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
# 256 x 151936 fp16 logits = 78 MB per chunk. The head runs at the peak of the step (every
# activation is still alive), so the chunk adds directly to the peak; 256 rows still keep the
# matmuls large enough for the tensor cores.
CHUNK_ROWS = 256
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


# ------------------------------------------- the same loss with the head weight on the CPU
#
# With the tied embedding / lm_head kept in pinned CPU memory (467 MB off the GPU), the head
# weight is streamed in vocabulary chunks, each copied once per step while the previous chunk
# computes. One pass is enough: per row the kernel keeps the online-softmax state (max, sum of
# exp, sum of exp*x for the entropy, argmax, target logit) across chunks, and for labelled rows it
# turns the chunk's logits into exp(x - running max), so that
#     acc = sum over chunks of exp(x - m) @ W_chunk     (rescaled by exp(m_old - m_new))
# gives softmax @ W = acc / sum_exp at the end; the hidden-state gradient is that minus W[target].

HEAD_CHUNK = 4096  # vocabulary rows per streamed chunk: 12.6 MB of fp16 weights


# col0 / n_cols / n_probs change per chunk and per batch: no divisibility specialization
@triton.jit(do_not_specialize=["n_cols", "col0", "n_probs"])
def soup_ce_stream_kernel(logits_ptr, stride_row, n_cols, col0, targets_ptr, max_ptr, sum_ptr,
                          dot_ptr, best_val_ptr, best_idx_ptr, x_target_ptr, rescale_ptr,
                          n_probs, BLOCK_V: tl.constexpr):
    """One row of one vocabulary chunk: update the row's online-softmax state; rows below
    `n_probs` also get their logits overwritten with exp(x - new max) (fp16, in [0, 1])."""
    row = tl.program_id(0)
    row_ptr = logits_ptr + row.to(tl.int64) * stride_row
    target = tl.load(targets_ptr + row)
    old_max = tl.load(max_ptr + row)
    run_max = old_max
    run_sum = tl.load(sum_ptr + row)
    run_dot = tl.load(dot_ptr + row)
    best_val = tl.load(best_val_ptr + row)
    best_idx = tl.load(best_idx_ptr + row)
    for start in range(0, n_cols, BLOCK_V):
        cols = start + tl.arange(0, BLOCK_V)
        valid = cols < n_cols
        x = tl.load(row_ptr + cols, mask=valid, other=float("-inf")).to(tl.float32)
        blk_max = tl.max(x, axis=0)
        blk_arg = tl.argmax(x, axis=0)
        best_idx = tl.where(blk_max > best_val, col0 + start + blk_arg, best_idx)
        best_val = tl.maximum(best_val, blk_max)
        new_max = tl.maximum(run_max, blk_max)
        rescale = tl.exp(run_max - new_max)
        e = tl.exp(x - new_max)
        run_sum = run_sum * rescale + tl.sum(e, axis=0)
        run_dot = run_dot * rescale + tl.sum(tl.where(valid, e * x, 0.0), axis=0)
        run_max = new_max
    tl.store(max_ptr + row, run_max)
    tl.store(sum_ptr + row, run_sum)
    tl.store(dot_ptr + row, run_dot)
    tl.store(best_val_ptr + row, best_val)
    tl.store(best_idx_ptr + row, best_idx)
    tl.store(rescale_ptr + row, tl.exp(old_max - run_max))
    local = target - col0
    if (local >= 0) & (local < n_cols):
        tl.store(x_target_ptr + row, tl.load(row_ptr + local).to(tl.float32))
    if row < n_probs:
        for start in range(0, n_cols, BLOCK_V):
            cols = start + tl.arange(0, BLOCK_V)
            valid = cols < n_cols
            x = tl.load(row_ptr + cols, mask=valid, other=float("-inf")).to(tl.float32)
            tl.store(row_ptr + cols, tl.exp(x - run_max).to(tl.float16), mask=valid)


class _HeadChunks:
    """Copies vocabulary chunks of a pinned CPU weight into two persistent GPU buffers, one chunk
    ahead of use. The buffers are allocated once on the main stream; the side stream only copies
    into a buffer after the main stream has recorded that it finished reading it."""

    def __init__(self, weight_cpu):
        self.weight, self.side = weight_cpu, _side_stream()
        self.spans = [(c, min(c + HEAD_CHUNK, weight_cpu.shape[0]))
                      for c in range(0, weight_cpu.shape[0], HEAD_CHUNK)]
        shape = (HEAD_CHUNK, weight_cpu.shape[1])
        if not _HEAD_RING or tuple(_HEAD_RING[0].shape) != shape:
            _HEAD_RING[:] = [torch.empty(shape, dtype=weight_cpu.dtype, device="cuda")
                             for _ in range(2)]
            _HEAD_FREE[:] = [None, None]

    def _copy(self, index):
        buffer = index % 2
        start, stop = self.spans[index]
        with torch.cuda.stream(self.side):
            if _HEAD_FREE[buffer] is not None:
                self.side.wait_event(_HEAD_FREE[buffer])
            _HEAD_RING[buffer][:stop - start].copy_(self.weight[start:stop], non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.side)
        return buffer, stop - start, done

    def __iter__(self):
        stream = torch.cuda.current_stream()
        self.side.wait_stream(stream)
        pending = self._copy(0)
        for index, (start, _stop) in enumerate(self.spans):
            buffer, rows, done = pending
            if index + 1 < len(self.spans):
                pending = self._copy(index + 1)
            stream.wait_event(done)
            yield start, _HEAD_RING[buffer][:rows]
            read = torch.cuda.Event()
            read.record(stream)
            _HEAD_FREE[buffer] = read


_HEAD_RING, _HEAD_FREE = [], []
_SIDE = []


def _side_stream():
    if not _SIDE:
        _SIDE.append(torch.cuda.Stream())
    return _SIDE[0]


class _StreamedHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight_cpu, rows, targets, n_labelled, denominator, with_grad):
        device = hidden.device
        n_rows = rows.numel()
        piece = hidden.index_select(0, rows).to(torch.float16)
        run_max = torch.full((n_rows,), float("-inf"), dtype=torch.float32, device=device)
        run_sum = torch.zeros(n_rows, dtype=torch.float32, device=device)
        run_dot = torch.zeros_like(run_sum)
        best_val = torch.full_like(run_max, float("-inf"))
        best_idx = torch.zeros(n_rows, dtype=torch.int32, device=device)
        x_target = torch.zeros_like(run_sum)
        rescale = torch.empty_like(run_sum)
        n_probs = n_labelled if with_grad else 0
        acc = torch.zeros((n_probs, hidden.shape[1]), dtype=torch.float32, device=device)
        for col0, chunk in _HeadChunks(weight_cpu):
            logits = piece @ chunk.t()
            soup_ce_stream_kernel[(n_rows,)](
                logits, logits.stride(0), chunk.shape[0], col0, targets, run_max, run_sum,
                run_dot, best_val, best_idx, x_target, rescale, n_probs, BLOCK_V=BLOCK_V,
                num_warps=8)
            if n_probs:
                acc.mul_(rescale[:n_probs, None]).add_((logits[:n_probs] @ chunk).float())
            del logits
        lse = run_max + torch.log(run_sum)
        entropy = lse - run_dot / run_sum
        has_target = targets != IGNORE_INDEX
        loss_rows = torch.where(has_target, lse - x_target, torch.zeros_like(lse))
        total = loss_rows.sum() / denominator
        grad_hidden = None
        if with_grad:
            grad_hidden = torch.zeros(hidden.shape, dtype=torch.float32, device=device)
            if n_probs:
                lab_targets = targets[:n_probs].long().cpu()
                w_target = weight_cpu.index_select(0, lab_targets).to(device, non_blocking=True)
                grad_rows = acc.div_(run_sum[:n_probs, None]).sub_(w_target.float())
                grad_hidden.index_copy_(0, rows[:n_probs], grad_rows.div_(denominator))
            ctx.save_for_backward(grad_hidden)
        ctx.with_grad, ctx.hidden_dtype = with_grad, hidden.dtype
        ctx.mark_non_differentiable(loss_rows, entropy, best_idx)
        return total, loss_rows, entropy, best_idx

    @staticmethod
    def backward(ctx, grad_total, *_unused):
        if not ctx.with_grad:
            return (None,) * 7
        (grad_hidden,) = ctx.saved_tensors
        grad = (grad_hidden * grad_total.float()).to(ctx.hidden_dtype)
        return grad, None, None, None, None, None, None


def fused_head_loss_cpu_weight(hidden, weight_cpu, labels, attention_mask, num_items_in_batch=None):
    """`fused_head_loss` with the head weight in pinned CPU memory, streamed in vocab chunks."""
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
        denominator = float(max(n_labelled, 1))
    elif torch.is_tensor(num_items_in_batch):
        denominator = float(num_items_in_batch.item())
    else:
        denominator = float(num_items_in_batch)
    with_grad = torch.is_grad_enabled() and hidden.requires_grad
    total, loss_rows, ent_rows, arg_rows = _StreamedHeadLoss.apply(
        flat, weight_cpu, rows, targets, n_labelled, denominator, with_grad)
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


def _pinned(weight):
    """The CPU head / embedding weight in pinned memory (once), for async chunk copies."""
    if not weight.data.is_pinned():
        from .stream import take_pinned

        pinned = take_pinned(weight)  # pinned right after loading (stream.start_prepack)
        weight.data = pinned if pinned is not None else weight.data.pin_memory()
    return weight.detach()


def _usable(model, labels):
    head = getattr(model, "lm_head", None)
    return (labels is not None and model.training and torch.is_grad_enabled()
            and isinstance(head, torch.nn.Linear) and head.bias is None
            and not head.weight.requires_grad
            and (head.weight.is_cuda or head.weight.dtype == torch.float16)
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
        weight = self.lm_head.weight
        if not weight.is_cuda:  # kept in pinned CPU memory (see prepare.py): stream it
            loss, _Last.step = fused_head_loss_cpu_weight(
                outputs.last_hidden_state, _pinned(weight), labels, attention_mask,
                kwargs.get("num_items_in_batch"))
        else:
            loss, _Last.step = fused_head_loss(outputs.last_hidden_state,
                                               _head_weight16(self.lm_head), labels,
                                               attention_mask,
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
