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


@triton.jit
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
        HEADS=triton.next_power_of_2(heads), INVERSE=inverse, num_warps=4)
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


def rope_half(q, k, cos, sin):
    cos, sin = cos.contiguous(), sin.contiguous()
    return _Rope.apply(q, cos, sin), _Rope.apply(k, cos, sin)


def _fits(q, k, cos, sin, unsqueeze_dim):
    dim = q.shape[-1]
    return (unsqueeze_dim == 1 and q.is_cuda and q.dim() == 4 and k.dim() == 4
            and cos.dim() == 3 and cos.shape == sin.shape and cos.shape[-1] == dim
            and cos.shape[1] == q.shape[2] and cos.shape[0] in (1, q.shape[0])
            and dim % 2 == 0 and (dim // 2) & (dim // 2 - 1) == 0
            and not cos.requires_grad and not sin.requires_grad and torch.is_grad_enabled()
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
