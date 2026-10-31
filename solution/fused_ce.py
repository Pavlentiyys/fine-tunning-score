"""Fused output head + cross-entropy + TRL/Soup metrics, without the [batch, seq, vocab] logits.

Base Soup builds fp16 logits for every position (2048 x 151936), upcasts them to fp32 for the
loss, then reads them again for TRL's entropy and accuracy and for Soup's per-row rewind loss:
several GB alive at the end of the forward pass, which is where the run peaks.

Here the head runs on chunks of rows. For each chunk one Triton kernel reads the fp16 logits row by
row and, in one pass over the vocabulary (online softmax), produces the log-sum-exp, the entropy
and the argmax; a second pass overwrites the logits with the gradient softmax - onehot, which the
chunk's matmul with the head weight turns into the gradient of the hidden states. The gradient is
taken in forward, so no logits outlive their chunk.

Only rows that need logits are computed: rows with a label (loss, accuracy, rewind) and other
non-padding rows (TRL's entropy covers every attended position). Padding rows are skipped.

Idea: chunked / fused linear cross-entropy as in Liger-Kernel (FusedLinearCrossEntropy) and Apple's
Cut Cross-Entropy (Wijmans et al., 2024); online softmax from Milakov & Gimelshein (2018). The code is
our own.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

IGNORE_INDEX = -100
CHUNK_ROWS = 1024  # 1024 x 151936 fp16 logits = 311 MB per chunk
BLOCK_V = 4096


@triton.jit
def soup_ce_rows_kernel(logits_ptr, stride_row, targets_ptr, loss_ptr, entropy_ptr, argmax_ptr,
                        V, WRITE_GRAD: tl.constexpr, BLOCK_V: tl.constexpr):
    """One program per row of fp16 logits: online log-sum-exp, entropy and argmax in one sweep;
    then (WRITE_GRAD) the row is overwritten with softmax - onehot(target), or zeros when the row
    has no target."""
    row = tl.program_id(0)
    row_ptr = logits_ptr + row.to(tl.int64) * stride_row
    target = tl.load(targets_ptr + row)

    run_max = tl.zeros((), dtype=tl.float32) - float("inf")
    run_sum = tl.zeros((), dtype=tl.float32)
    run_dot = tl.zeros((), dtype=tl.float32)  # sum exp(x - max) * x, for the entropy
    best_val = tl.zeros((), dtype=tl.float32) - float("inf")
    best_idx = tl.zeros((), dtype=tl.int32)
    for start in range(0, V, BLOCK_V):
        cols = start + tl.arange(0, BLOCK_V)
        valid = cols < V
        x = tl.load(row_ptr + cols, mask=valid, other=float("-inf")).to(tl.float32)
        blk_max = tl.max(x, axis=0)
        blk_arg = tl.argmax(x, axis=0)
        # strict > keeps the first index on ties, as torch.argmax does
        best_idx = tl.where(blk_max > best_val, start + blk_arg, best_idx)
        best_val = tl.maximum(best_val, blk_max)
        new_max = tl.maximum(run_max, blk_max)
        rescale = tl.exp(run_max - new_max)
        e = tl.exp(x - new_max)
        run_sum = run_sum * rescale + tl.sum(e, axis=0)
        run_dot = run_dot * rescale + tl.sum(tl.where(valid, e * x, 0.0), axis=0)
        run_max = new_max

    lse = run_max + tl.log(run_sum)
    tl.store(entropy_ptr + row, lse - run_dot / run_sum)
    tl.store(argmax_ptr + row, best_idx)
    has_target = target != -100
    safe_target = tl.where(has_target, target, 0)
    x_target = tl.load(row_ptr + safe_target).to(tl.float32)
    tl.store(loss_ptr + row, tl.where(has_target, lse - x_target, 0.0))

    if WRITE_GRAD:
        for start in range(0, V, BLOCK_V):
            cols = start + tl.arange(0, BLOCK_V)
            valid = cols < V
            x = tl.load(row_ptr + cols, mask=valid, other=0.0).to(tl.float32)
            g = tl.exp(x - lse) - tl.where(cols == target, 1.0, 0.0)
            g = tl.where(has_target, g, 0.0)
            tl.store(row_ptr + cols, g.to(tl.float16), mask=valid)


def ce_rows(logits, targets, write_grad):
    """Per-row (loss, entropy, argmax) of fp16 `logits` [n, V]; with `write_grad`, `logits` is
    overwritten with softmax - onehot (zero rows where target is -100)."""
    n, vocab = logits.shape
    loss = torch.empty(n, dtype=torch.float32, device=logits.device)
    entropy = torch.empty_like(loss)
    argmax = torch.empty(n, dtype=torch.int32, device=logits.device)
    if n:
        soup_ce_rows_kernel[(n,)](logits, logits.stride(0), targets, loss, entropy, argmax, vocab,
                                  WRITE_GRAD=write_grad, BLOCK_V=BLOCK_V, num_warps=8)
    return loss, entropy, argmax


@dataclass(frozen=True)
class HeadStep:
    """What the trainer's metrics need from a step without logits; `token_loss` follows Soup's
    chunked-step convention, so the rewind recorder reads its per-row losses from it."""

    token_entropy: torch.Tensor  # [B, T] float32, 0 at padding
    token_loss: torch.Tensor  # [B, T] float32: position t scored against labels[:, t + 1]
    correct: torch.Tensor  # number of labelled positions whose argmax is the label
    labelled: torch.Tensor  # number of labelled positions


class _FusedHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight16, rows, targets, n_labelled, denominator, with_grad):
        # hidden [N, H] (all positions), rows: the positions to score, labelled ones first
        n_rows = rows.numel()
        loss_rows = torch.empty(n_rows, dtype=torch.float32, device=hidden.device)
        ent_rows = torch.empty_like(loss_rows)
        arg_rows = torch.empty(n_rows, dtype=torch.int32, device=hidden.device)
        # fp32 even for fp16 hidden states: 1/denominator-sized values would lose bits in fp16
        # before the loss scale reaches them in backward
        grad_hidden = torch.zeros(hidden.shape, dtype=torch.float32,
                                  device=hidden.device) if with_grad else None
        for start in range(0, n_rows, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, n_rows)
            idx = rows[start:stop]
            piece = hidden.index_select(0, idx).to(torch.float16)
            logits = piece @ weight16.t()
            del piece
            need_grad = with_grad and start < n_labelled
            loss, ent, arg = ce_rows(logits, targets[start:stop], need_grad)
            loss_rows[start:stop], ent_rows[start:stop], arg_rows[start:stop] = loss, ent, arg
            if need_grad:
                # softmax - onehot is in [-1, 1]: fp16 keeps it; the 1/denominator is applied in
                # fp32 afterwards so small probabilities do not underflow
                grad_hidden.index_copy_(0, idx, (logits @ weight16).float().div_(denominator))
            del logits
        total = loss_rows.sum() / denominator
        if with_grad:
            ctx.save_for_backward(grad_hidden)
        ctx.with_grad, ctx.hidden_dtype = with_grad, hidden.dtype
        ctx.mark_non_differentiable(loss_rows, ent_rows, arg_rows)
        return total, loss_rows, ent_rows, arg_rows

    @staticmethod
    def backward(ctx, grad_total, *_unused):
        if not ctx.with_grad:
            return (None,) * 7
        (grad_hidden,) = ctx.saved_tensors
        grad = (grad_hidden * grad_total.float()).to(ctx.hidden_dtype)
        return grad, None, None, None, None, None, None


def fused_head_loss(hidden, weight16, labels, attention_mask, num_items_in_batch=None):
    """`ForCausalLMLoss(lm_head(hidden), labels)` (the sum over labelled positions divided by
    `num_items_in_batch`, or their mean) and a HeadStep, without the full logits."""
    batch, seq, width = hidden.shape
    flat = hidden.reshape(batch * seq, width)
    target = torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[:, 1:].reshape(-1)
    target = target.to(hidden.device)
    labelled_mask = target != IGNORE_INDEX
    if attention_mask is not None:
        attended = attention_mask.reshape(-1).to(hidden.device).bool()
    else:
        attended = torch.ones_like(labelled_mask)
    lab_rows = labelled_mask.nonzero().squeeze(1)
    other_rows = (attended & ~labelled_mask).nonzero().squeeze(1)
    rows = torch.cat([lab_rows, other_rows])
    n_labelled = lab_rows.numel()
    targets = target.index_select(0, rows).to(torch.int32).contiguous()
    labelled = torch.tensor(n_labelled, device=hidden.device)
    if num_items_in_batch is None:
        denominator = float(max(n_labelled, 1)) if n_labelled else float("nan")
    elif torch.is_tensor(num_items_in_batch):
        denominator = float(num_items_in_batch.item())
    else:
        denominator = float(num_items_in_batch)
    with_grad = torch.is_grad_enabled() and hidden.requires_grad
    total, loss_rows, ent_rows, arg_rows = _FusedHeadLoss.apply(
        flat, weight16, rows, targets, n_labelled, denominator, with_grad)

    with torch.no_grad():
        token_loss = torch.zeros(batch * seq, dtype=torch.float32, device=hidden.device)
        token_loss[lab_rows] = loss_rows[:n_labelled]
        token_entropy = torch.zeros(batch * seq, dtype=torch.float32, device=hidden.device)
        token_entropy[rows] = ent_rows
        correct = (arg_rows[:n_labelled].long() == target[lab_rows]).sum()
    step = HeadStep(token_entropy=token_entropy.view(batch, seq),
                    token_loss=token_loss.view(batch, seq), correct=correct, labelled=labelled)
    return total, step


# ---------------------------------------------------------------- wiring into transformers/TRL

class _Last:
    step = None


_W16_CACHE = {}


def _head_weight16(lm_head):
    """The frozen head weight in fp16, cast once (base Soup's autocast casts it every step)."""
    weight = lm_head.weight
    key = (weight.data_ptr(), weight.dtype, tuple(weight.shape))
    cached = _W16_CACHE.get(id(lm_head))
    if cached is None or cached[0] != key:
        w16 = weight.detach() if weight.dtype == torch.float16 else weight.detach().to(torch.float16)
        _W16_CACHE.clear()
        _W16_CACHE[id(lm_head)] = cached = (key, w16)
    return cached[1]


def _usable(model, labels):
    head = getattr(model, "lm_head", None)
    return (labels is not None and model.training and torch.is_grad_enabled()
            and isinstance(head, torch.nn.Linear) and head.bias is None
            and not head.weight.requires_grad and head.weight.is_cuda
            and getattr(model.config, "final_logit_softcapping", None) is None)


def patch_qwen2_causal_lm():
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2ForCausalLM.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
        if not _usable(self, labels) or logits_to_keep not in (0, None):
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                            logits_to_keep=logits_to_keep, **kwargs)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        loss, _Last.step = fused_head_loss(outputs.last_hidden_state, _head_weight16(self.lm_head),
                                           labels, attention_mask,
                                           kwargs.get("num_items_in_batch"))
        return CausalLMOutputWithPast(loss=loss, logits=None,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions)

    modeling_qwen2.Qwen2ForCausalLM.forward = forward


def patch_sft_trainer():
    """TRL's `compute_loss` reads `outputs.logits` for `entropy` and `mean_token_accuracy`; log the
    same columns by TRL's formulas from the HeadStep, and leave the step where Soup's rewind
    recorder looks for a chunked step's per-token losses."""
    import transformers
    import trl

    original = trl.SFTTrainer.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        args = getattr(self, "args", None)
        if (not isinstance(model, torch.nn.Module) or not isinstance(inputs, dict)
                or "labels" not in inputs or "shift_labels" in inputs
                or inputs.get("attention_mask") is None
                or getattr(args, "use_liger_kernel", False)
                or getattr(self, "num_virtual_tokens", 0)
                or getattr(self, "aux_loss_enabled", False)
                or not getattr(model, "training", False)):
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        mode = "train" if self.model.training else "eval"
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        attention_mask = inputs.get("attention_mask")
        _Last.step = None
        loss, outputs = transformers.Trainer.compute_loss(
            self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        step, _Last.step = _Last.step, None
        if step is None:
            raise RuntimeError("fused head loss did not run on a training forward")
        self._soup_chunked_step = step  # soup_cli.utils.stream_loss_chunk.STEP_ATTR
        gather = self.accelerator.gather_for_metrics
        with torch.no_grad():
            entropy = (step.token_entropy * attention_mask).sum() / attention_mask.sum()
            self._metrics[mode]["entropy"].append(gather(entropy).mean().item())
            if mode == "train":
                self._total_train_tokens += gather(attention_mask.sum()).sum().item()
            self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
            correct, total = gather(step.correct).sum(), gather(step.labelled).sum()
            accuracy = (correct / total).item() if total > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)
        return (loss, outputs) if return_outputs else loss

    trl.SFTTrainer.compute_loss = compute_loss
