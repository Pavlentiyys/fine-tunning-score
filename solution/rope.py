"""Rotary position embedding for q and k in one Triton kernel, fp16 out.

In base Soup `apply_rotary_pos_emb` multiplies the fp16 q/k by fp32 cos/sin tables: six elementwise
kernels (two products, a cat for rotate_half, a negation, an add) per tensor, an fp32 result, and
then SDPA under autocast makes fp16 copies of it. Here one program per (batch, position) rotates all
heads of that position in fp32 registers and stores fp16, the dtype SDPA computes in anyway. The
backward is the inverse rotation (the same kernel with the sine negated), so nothing is saved but
the cos/sin tables, which the model shares across layers.
"""

import torch
import triton
import triton.language as tl

from .mode import fast_ok

# q (12 heads) and k (2 heads) share one compiled kernel; the extra lanes are masked
ROPE_HEADS = 16


# `seq` changes with every batch: no divisibility specialization, or each new length class would
# compile again (every run starts with an empty Triton cache and the compile time is measured)
@triton.jit(do_not_specialize=["seq", "sxb", "sob", "scb"])
def soup_rope_kernel(x_ptr, out_ptr, cos_ptr, sin_ptr,
                     sxb, sxh, sxt, sob, soh, sot, scb, sct, seq, n_heads,
                     HALF: tl.constexpr, HEADS: tl.constexpr, INVERSE: tl.constexpr):
    """out[:half] = x1*c1 - x2*s1, out[half:] = x2*c2 + x1*s2 (INVERSE: with -s2 and -s1 swapped
    in, which undoes the rotation, i.e. the gradient)."""
    pid = tl.program_id(0)
    b = pid // seq
    t = pid % seq
    h = tl.arange(0, HEADS)[:, None]
    d = tl.arange(0, HALF)[None, :]
    live = h < n_heads
    x_at = x_ptr + b.to(tl.int64) * sxb + t.to(tl.int64) * sxt + h * sxh + d
    o_at = out_ptr + b.to(tl.int64) * sob + t.to(tl.int64) * sot + h * soh + d
    tab = b.to(tl.int64) * scb + t.to(tl.int64) * sct + d
    c1 = tl.load(cos_ptr + tab).to(tl.float32)
    c2 = tl.load(cos_ptr + tab + HALF).to(tl.float32)
    s1 = tl.load(sin_ptr + tab).to(tl.float32)
    s2 = tl.load(sin_ptr + tab + HALF).to(tl.float32)
    if INVERSE:
        neg_s2 = -s2
        s2 = -s1
        s1 = neg_s2
    x1 = tl.load(x_at, mask=live, other=0.0).to(tl.float32)
    x2 = tl.load(x_at + HALF, mask=live, other=0.0).to(tl.float32)
    tl.store(o_at, (x1 * c1 - x2 * s1).to(out_ptr.dtype.element_ty), mask=live)
    tl.store(o_at + HALF, (x2 * c2 + x1 * s2).to(out_ptr.dtype.element_ty), mask=live)


def _rotate(x, cos, sin, out_dtype, inverse):
    """x [B, H, T, D] (any strides, unit stride on D); cos/sin [B or 1, T, D]. The output is laid
    out like a q_proj output viewed as [B, T, H, D] and transposed, as in base Soup."""
    batch, heads, seq, dim = x.shape
    if x.stride(-1) != 1:
        x = x.contiguous()
    out = torch.empty((batch, seq, heads, dim), dtype=out_dtype, device=x.device).transpose(1, 2)
    scb = cos.stride(0) if cos.shape[0] > 1 else 0
    soup_rope_kernel[(batch * seq,)](
        x, out, cos, sin, x.stride(0), x.stride(1), x.stride(2), out.stride(0), out.stride(1),
        out.stride(2), scb, cos.stride(1), seq, heads, HALF=dim // 2,
        HEADS=ROPE_HEADS if heads <= ROPE_HEADS else triton.next_power_of_2(heads), INVERSE=inverse, num_warps=4)
    return out


class _Rope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        ctx.x_dtype = x.dtype
        return _rotate(x, cos, sin, torch.float16, inverse=False)

    @staticmethod
    def backward(ctx, grad):
        cos, sin = ctx.saved_tensors
        return _rotate(grad, cos, sin, ctx.x_dtype, inverse=True), None, None


@triton.jit(do_not_specialize=["seq", "sqb", "skb", "soqb", "sokb", "scb"])
def soup_rope_qk_kernel(q_ptr, k_ptr, oq_ptr, ok_ptr, cos_ptr, sin_ptr,
                        sqb, sqh, sqt, skb, skh, skt, soqb, soqh, soqt, sokb, sokh, sokt,
                        scb, sct, seq, n_q, n_k,
                        HALF: tl.constexpr, QH: tl.constexpr, KH: tl.constexpr,
                        INVERSE: tl.constexpr):
    """soup_rope_kernel's rotation for q and k of the same positions in one launch: one program
    per (batch, position) loads the cos/sin row once and rotates all q heads and all k heads."""
    pid = tl.program_id(0)
    b = (pid // seq).to(tl.int64)
    t = (pid % seq).to(tl.int64)
    d = tl.arange(0, HALF)[None, :]
    tab = b * scb + t * sct + d
    c1 = tl.load(cos_ptr + tab).to(tl.float32)
    c2 = tl.load(cos_ptr + tab + HALF).to(tl.float32)
    s1 = tl.load(sin_ptr + tab).to(tl.float32)
    s2 = tl.load(sin_ptr + tab + HALF).to(tl.float32)
    if INVERSE:
        neg_s2 = -s2
        s2 = -s1
        s1 = neg_s2
    hq = tl.arange(0, QH)[:, None]
    live_q = hq < n_q
    xq = q_ptr + b * sqb + t * sqt + hq * sqh + d
    oq = oq_ptr + b * soqb + t * soqt + hq * soqh + d
    q1 = tl.load(xq, mask=live_q, other=0.0).to(tl.float32)
    q2 = tl.load(xq + HALF, mask=live_q, other=0.0).to(tl.float32)
    tl.store(oq, (q1 * c1 - q2 * s1).to(oq_ptr.dtype.element_ty), mask=live_q)
    tl.store(oq + HALF, (q2 * c2 + q1 * s2).to(oq_ptr.dtype.element_ty), mask=live_q)
    hk = tl.arange(0, KH)[:, None]
    live_k = hk < n_k
    xk = k_ptr + b * skb + t * skt + hk * skh + d
    ok = ok_ptr + b * sokb + t * sokt + hk * sokh + d
    k1 = tl.load(xk, mask=live_k, other=0.0).to(tl.float32)
    k2 = tl.load(xk + HALF, mask=live_k, other=0.0).to(tl.float32)
    tl.store(ok, (k1 * c1 - k2 * s1).to(ok_ptr.dtype.element_ty), mask=live_k)
    tl.store(ok + HALF, (k2 * c2 + k1 * s2).to(ok_ptr.dtype.element_ty), mask=live_k)


def heads_lanes(heads):
    return max(2, triton.next_power_of_2(heads))


def _rotate_qk(q, k, cos, sin, out_dtype, inverse):
    """q [B, Hq, T, D], k [B, Hk, T, D] (unit stride on D); outputs laid out as _rotate's."""
    batch, q_heads, seq, dim = q.shape
    k_heads = k.shape[1]
    if q.stride(-1) != 1:
        q = q.contiguous()
    if k.stride(-1) != 1:
        k = k.contiguous()
    out_q = torch.empty((batch, seq, q_heads, dim), dtype=out_dtype, device=q.device).transpose(1, 2)
    out_k = torch.empty((batch, seq, k_heads, dim), dtype=out_dtype, device=k.device).transpose(1, 2)
    scb = cos.stride(0) if cos.shape[0] > 1 else 0
    soup_rope_qk_kernel[(batch * seq,)](
        q, k, out_q, out_k, cos, sin, q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), out_q.stride(0), out_q.stride(1), out_q.stride(2),
        out_k.stride(0), out_k.stride(1), out_k.stride(2), scb, cos.stride(1), seq, q_heads,
        k_heads, HALF=dim // 2, QH=heads_lanes(q_heads), KH=heads_lanes(k_heads),
        INVERSE=inverse, num_warps=4)
    return out_q, out_k


class _RopeQK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        # the cos/sin tables are shared by all layers and never modified: kept as attributes, not
        # saved tensors, so q's and k's gradients may also come from two separate backward calls
        # (one node serves both outputs; saved tensors would be freed by the first call)
        ctx.cos, ctx.sin = cos, sin
        ctx.dtypes = q.dtype, k.dtype
        return _rotate_qk(q, k, cos, sin, torch.float16, inverse=False)

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        cos, sin = ctx.cos, ctx.sin
        if ctx.dtypes[0] != ctx.dtypes[1]:  # one launch needs one output dtype
            return (_rotate(grad_q, cos, sin, ctx.dtypes[0], inverse=True),
                    _rotate(grad_k, cos, sin, ctx.dtypes[1], inverse=True), None, None)
        dq, dk = _rotate_qk(grad_q, grad_k, cos, sin, ctx.dtypes[0], inverse=True)
        return dq, dk, None, None


def rope_half(q, k, cos, sin):
    cos, sin = cos.contiguous(), sin.contiguous()
    if (q.shape[0] == k.shape[0] and q.shape[2] == k.shape[2] and q.shape[3] == k.shape[3]
            and q.device == k.device):
        return _RopeQK.apply(q, k, cos, sin)
    return _Rope.apply(q, cos, sin), _Rope.apply(k, cos, sin)


def _fits(q, k, cos, sin, unsqueeze_dim):
    dim = q.shape[-1]
    return (unsqueeze_dim == 1 and q.is_cuda and q.dim() == 4 and k.dim() == 4
            and cos.dim() == 3 and cos.shape == sin.shape and cos.shape[-1] == dim
            and cos.shape[1] == q.shape[2] and cos.shape[0] in (1, q.shape[0])
            and dim % 2 == 0 and (dim // 2) & (dim // 2 - 1) == 0
            and not cos.requires_grad and not sin.requires_grad and fast_ok()
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.float16)


def patch_qwen2_rope():
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.apply_rotary_pos_emb

    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        if _fits(q, k, cos, sin, unsqueeze_dim):
            return rope_half(q, k, cos, sin)
        return original(q, k, cos, sin, unsqueeze_dim)

    modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb
