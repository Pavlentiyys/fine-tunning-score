"""Run the token-wise part of Qwen2 on the real tokens only, and keep the tied embedding in fp16.

Padding. TRL pads each batch of 4 rows to its longest row. The rows average ~220 tokens (median
140) against a longest of ~350-512, so ~40% of the positions are padding, and base Soup runs the
whole model on them. Every op except the attention itself is token-wise: the norms, the residual
adds, the q/k/v/o projections with their LoRA, and the MLP. Here the residual stream lives as
[real tokens, hidden] and all of those run on it; only RoPE and SDPA see the padded
[batch, heads, seq, head_dim] layout, with the same causal + padding mask and RoPE tables as base
Soup: q, k and v are scattered into it (zeros at the padding, which the mask hides from every real
query) and the attention output is gathered back before o_proj. Removing padding inside a batch is
allowed by the rules; no rows are mixed across batches. (LoRA's dropout now draws its mask over the
real tokens only, so the masks are not bit-identical to base Soup's; they are the same dropout.)

Embedding. `prepare_model_for_kbit_training` upcasts the tied embedding / lm_head (151936 x 1536) to
fp32: 934 MB. The checkpoint stores it in bf16, whose values fp16 holds exactly in the range they
use, and the head already runs in fp16 under autocast. Storing it in fp16 halves it; the embedding
output is upcast to fp32 so the residual stream stays fp32 as in base Soup.
"""

import torch
from torch.utils.checkpoint import checkpoint

from .lora import draw_seeds, replay_seeds
from .stream import _REGISTRY, get_manager, restore_all, streamed_layer

# Keep only each decoder layer's input for the backward and recompute the layer there (our own
# recomputation: Soup's gradient_checkpointing switch stays off). Trades one more forward per layer
# for the activations of all 28 layers.
RECOMPUTE_LAYERS = True
SEEDS_PER_LAYER = 4  # LoRA dropout seeds a layer may draw (q_proj and v_proj use two)

_EMBED_DONE = set()
_STEPS = [0]


def _mem(tag):
    """One line in train.log: where the allocator's peaks come from (read only)."""
    mib = 2**20
    print(f"[mem] {tag}: allocated {torch.cuda.memory_allocated() / mib:.0f}, "
          f"max allocated {torch.cuda.max_memory_allocated() / mib:.0f}, "
          f"max reserved {torch.cuda.max_memory_reserved() / mib:.0f} MiB", flush=True)


def _embedding_to_fp16(model):
    """Once: the frozen (tied) input embedding in fp16. The fp32 weight goes through the CPU, so
    the fp32 and fp16 copies are never on the GPU together: that moment lasted milliseconds, was
    the allocator's peak, and nvidia-smi (100 ms samples) only sometimes saw it."""
    weight = model.embed_tokens.weight
    if id(weight) in _EMBED_DONE or weight.dtype != torch.float32 or weight.requires_grad:
        return
    _mem("before embedding to fp16")
    half = weight.data.cpu().half()
    weight.data = torch.empty(0, dtype=torch.float16, device=weight.device)
    torch.cuda.empty_cache()
    weight.data = half.to(weight.device)
    _EMBED_DONE.add(id(weight))
    torch.cuda.empty_cache()
    _mem("after embedding to fp16")


def _embedding_to_gpu(model):
    """The embedding kept on the CPU goes back to the GPU for any forward that is not ours."""
    weight = model.embed_tokens.weight
    if not weight.is_cuda and torch.cuda.is_available():
        weight.data = weight.data.to("cuda")


def _usable(model, input_ids, attention_mask, past_key_values, inputs_embeds, use_cache, kwargs):
    return (model.training and torch.is_grad_enabled() and input_ids is not None
            and inputs_embeds is None and past_key_values is None and not use_cache
            and torch.is_tensor(attention_mask) and attention_mask.dim() == 2
            and input_ids.is_cuda and not kwargs.get("output_hidden_states")
            and not kwargs.get("output_attentions"))


def _attention(attn, normed, rows, flat_len, shape, mask, position_embeddings, kwargs):
    """Qwen2Attention.forward with the projections on the real tokens only."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen2 import modeling_qwen2

    batch, seq = shape

    def to_heads(states):
        padded = states.new_zeros(flat_len, states.shape[-1]).index_copy(0, rows, states)
        return padded.view(batch, seq, -1, attn.head_dim).transpose(1, 2)

    query = to_heads(attn.q_proj(normed))
    key = to_heads(attn.k_proj(normed))
    value = to_heads(attn.v_proj(normed))
    cos, sin = position_embeddings
    query, key = modeling_qwen2.apply_rotary_pos_emb(query, key, cos, sin)
    interface = ALL_ATTENTION_FUNCTIONS.get_interface(attn.config._attn_implementation,
                                                      modeling_qwen2.eager_attention_forward)
    out, _ = interface(attn, query, key, value, mask,
                       dropout=0.0 if not attn.training else attn.attention_dropout,
                       scaling=attn.scaling, sliding_window=attn.sliding_window, **kwargs)
    out = out.reshape(flat_len, -1).index_select(0, rows)
    return attn.o_proj(out)


def _decoder_layer(layer, hidden, rows, flat_len, shape, mask, position_embeddings, kwargs):
    """Qwen2DecoderLayer.forward with the residual stream as [n_real, hidden]."""
    residual = hidden
    normed = layer.input_layernorm(hidden)
    hidden = residual + _attention(layer.self_attn, normed, rows, flat_len, shape, mask,
                                   position_embeddings, kwargs)
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
        if _REGISTRY and not self.training:
            restore_all()  # anything that is not a training forward needs the weights on the GPU
        if not _usable(self, input_ids, attention_mask, past_key_values, inputs_embeds, use_cache,
                       kwargs):
            if _REGISTRY:
                restore_all()
            _embedding_to_gpu(self)
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        _embedding_to_fp16(self)
        _STEPS[0] += 1
        if _STEPS[0] in (1, 2, 3, 10):
            _mem(f"forward call {_STEPS[0]}")
        batch, seq = input_ids.shape
        flat_len = batch * seq
        rows = attention_mask.reshape(-1).nonzero().squeeze(1)
        weight = self.embed_tokens.weight
        if weight.is_cuda:
            embeds = self.embed_tokens(input_ids)
            if embeds.dtype == torch.float16:
                embeds = embeds.float()  # the residual stream stays fp32, as in base Soup
            hidden = embeds.reshape(flat_len, -1).index_select(0, rows)
        else:
            # embedding in pinned CPU memory (prepare.py): look up the real tokens' rows there and
            # copy only those; `embeds` is a zero-stride stand-in for the mask / RoPE helpers,
            # which read only its shape, dtype and device
            from .fused_ce import _pinned

            ids = input_ids.reshape(-1).index_select(0, rows).cpu()
            hidden = _pinned(weight).index_select(0, ids).to(input_ids.device).float()
            embeds = torch.zeros((), dtype=torch.float32, device=input_ids.device).expand(
                batch, seq, weight.shape[1])
        if position_ids is None:
            position_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0)
        mask_kwargs = {"config": self.config, "inputs_embeds": embeds,
                       "attention_mask": attention_mask, "past_key_values": None,
                       "position_ids": position_ids}
        masks = {"full_attention": create_causal_mask(**mask_kwargs)}
        if self.has_sliding_layers:
            masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        position_embeddings = self.rotary_emb(embeds, position_ids)

        layers = self.layers[: self.config.num_hidden_layers]
        manager = get_manager(layers) if self.training else None
        for i, layer in enumerate(layers):
            args = (layer, rows, flat_len, (batch, seq), masks[self.config.layer_types[i]],
                    position_embeddings, kwargs)
            if manager is not None or RECOMPUTE_LAYERS:
                seeds = draw_seeds(SEEDS_PER_LAYER)

                def run(states, args=args, seeds=seeds):
                    with replay_seeds(seeds):
                        return _decoder_layer(args[0], states, *args[1:])

                if manager is not None:
                    hidden = streamed_layer(manager, i, run, layer, hidden)
                else:
                    hidden = checkpoint(run, hidden, use_reentrant=False,
                                        preserve_rng_state=False)
            else:
                hidden = _decoder_layer(layer, hidden, *args[1:])
        hidden = self.norm(hidden)
        out = hidden.new_zeros(flat_len, hidden.shape[-1]).index_copy(0, rows, hidden)
        return BaseModelOutputWithPast(last_hidden_state=out.view(batch, seq, -1),
                                       past_key_values=None)

    modeling_qwen2.Qwen2Model.forward = forward
