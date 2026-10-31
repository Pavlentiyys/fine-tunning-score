"""Run the token-wise part of Qwen2 on the real tokens only, and keep the tied embedding in fp16.

Padding. TRL pads each batch of 4 rows to its longest row. The rows average ~220 tokens (median
140) against a longest of ~350-512, so ~40% of the positions are padding, and base Soup runs the
whole model on them. Every op except attention is token-wise: the norms, the residual adds and the
MLP (which holds ~88% of the matmul FLOPs). Here the residual stream lives as [real tokens, hidden];
only the attention block sees the padded [batch, seq, hidden] layout: its input is scattered back
(zeros at the padding) and its output gathered again. The attention module itself is untouched and
gets the same padded shape, mask and RoPE tables as in base Soup, so even LoRA's dropout masks on
q_proj / v_proj are drawn exactly as before. Removing padding inside a batch is allowed by the rules;
no rows are mixed across batches.

Embedding. `prepare_model_for_kbit_training` upcasts the tied embedding / lm_head (151936 x 1536) to
fp32: 934 MB. The checkpoint stores it in bf16, whose values fp16 holds exactly in the range they
use, and the head already runs in fp16 under autocast. Storing it in fp16 halves it; the embedding
output is upcast to fp32 so the residual stream stays fp32 as in base Soup.
"""

import torch

_EMBED_DONE = set()


def _embedding_to_fp16(model):
    """Once: the frozen (tied) input embedding in fp16; the freed fp32 copy goes back to the
    allocator before the activations of the first step need room."""
    weight = model.embed_tokens.weight
    if id(weight) in _EMBED_DONE or weight.dtype != torch.float32 or weight.requires_grad:
        return
    weight.data = weight.data.to(torch.float16)
    _EMBED_DONE.add(id(weight))
    torch.cuda.empty_cache()


def _usable(model, input_ids, attention_mask, past_key_values, inputs_embeds, use_cache, kwargs):
    return (model.training and torch.is_grad_enabled() and input_ids is not None
            and inputs_embeds is None and past_key_values is None and not use_cache
            and torch.is_tensor(attention_mask) and attention_mask.dim() == 2
            and input_ids.is_cuda and not kwargs.get("output_hidden_states")
            and not kwargs.get("output_attentions"))


def _decoder_layer(layer, hidden, rows, flat_len, shape, mask, position_ids, position_embeddings,
                   kwargs):
    """Qwen2DecoderLayer.forward with the residual stream as [n_real, hidden]."""
    residual = hidden
    normed = layer.input_layernorm(hidden)
    padded = normed.new_zeros(flat_len, normed.shape[-1]).index_copy(0, rows, normed)
    attn, _ = layer.self_attn(hidden_states=padded.view(*shape, -1), attention_mask=mask,
                              position_ids=position_ids, past_key_values=None, use_cache=False,
                              position_embeddings=position_embeddings, **kwargs)
    hidden = residual + attn.reshape(flat_len, -1).index_select(0, rows)
    residual = hidden
    hidden = layer.mlp(layer.post_attention_layernorm(hidden))
    return residual + hidden


def patch_qwen2_model():
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2Model.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        if not _usable(self, input_ids, attention_mask, past_key_values, inputs_embeds, use_cache,
                       kwargs):
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        _embedding_to_fp16(self)
        batch, seq = input_ids.shape
        embeds = self.embed_tokens(input_ids)
        if embeds.dtype == torch.float16:
            embeds = embeds.float()  # the residual stream stays fp32, as in base Soup
        if position_ids is None:
            position_ids = torch.arange(seq, device=embeds.device).unsqueeze(0)
        mask_kwargs = {"config": self.config, "inputs_embeds": embeds,
                       "attention_mask": attention_mask, "past_key_values": None,
                       "position_ids": position_ids}
        masks = {"full_attention": create_causal_mask(**mask_kwargs)}
        if self.has_sliding_layers:
            masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        position_embeddings = self.rotary_emb(embeds, position_ids)

        flat_len = batch * seq
        rows = attention_mask.reshape(-1).nonzero().squeeze(1)
        hidden = embeds.reshape(flat_len, -1).index_select(0, rows)
        for i, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden = _decoder_layer(layer, hidden, rows, flat_len, (batch, seq),
                                    masks[self.config.layer_types[i]], position_ids,
                                    position_embeddings, kwargs)
        hidden = self.norm(hidden)
        out = hidden.new_zeros(flat_len, hidden.shape[-1]).index_copy(0, rows, hidden)
        return BaseModelOutputWithPast(last_hidden_state=out.view(batch, seq, -1),
                                       past_key_values=None)

    modeling_qwen2.Qwen2Model.forward = forward
