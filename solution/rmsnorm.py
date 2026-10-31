"""Fused RMSNorm for Qwen2 with an fp16 output, forward and backward in Triton.

In base Soup the residual stream is fp32 (`prepare_model_for_kbit_training` keeps the embedding and
the norms in fp32), so `Qwen2RMSNorm` returns fp32: the q/k/v/gate/up projections then get an fp32
input, bitsandbytes' `Linear4bit` casts it to fp16 for the matmul and casts the result back to the
input's dtype, and every activation of the layer is stored in fp32. Under fp16 autocast the next
matmul rounds the norm's output to fp16 anyway, so returning fp16 here changes no matmul input and
halves the activations that follow (q, k, v, gate, up, the SwiGLU product).

One program per row: the sum of squares, the rsqrt and the weight in fp32, fp16 stores. For the
backward (the norm weight is frozen under LoRA, so only the input gradient) the forward also stores
the normalized row xhat = x * rstd in fp16 (bounded by sqrt(hidden), so fp16 keeps it to 1e-3
relative) plus one fp32 rstd per row; the fp32 residual itself is not kept alive by the norm.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def soup_rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, xhat_ptr, rstd_ptr, n_cols, eps,
                            BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    inside = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=inside, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    w = tl.load(w_ptr + cols, mask=inside, other=0.0).to(tl.float32)
    xhat = x * rstd
    tl.store(y_ptr + row * n_cols + cols, (xhat * w).to(y_ptr.dtype.element_ty), mask=inside)
    tl.store(xhat_ptr + row * n_cols + cols, xhat.to(tl.float16), mask=inside)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def soup_rmsnorm_bwd_kernel(dy_ptr, xhat_ptr, w_ptr, rstd_ptr, dx_ptr, n_cols, BLOCK: tl.constexpr):
    """dx = rstd * (w*dy - xhat * mean(w*dy * xhat)), from the saved fp16 xhat = x * rstd."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    inside = cols < n_cols
    dy = tl.load(dy_ptr + row * n_cols + cols, mask=inside, other=0.0).to(tl.float32)
    xhat = tl.load(xhat_ptr + row * n_cols + cols, mask=inside, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=inside, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    wdy = w * dy
    proj = tl.sum(wdy * xhat, axis=0) / n_cols
    dx = (wdy - xhat * proj) * rstd
    tl.store(dx_ptr + row * n_cols + cols, dx.to(dx_ptr.dtype.element_ty), mask=inside)


class _RMSNormHalfOut(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, out_dtype):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        rows, cols = x2.shape
        y = torch.empty((rows, cols), dtype=out_dtype, device=x.device)
        xhat = torch.empty((rows, cols), dtype=torch.float16, device=x.device)
        rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
        block = triton.next_power_of_2(cols)
        warps = 8 if block >= 2048 else 4
        if rows:
            soup_rmsnorm_fwd_kernel[(rows,)](x2, weight, y, xhat, rstd, cols, eps, BLOCK=block,
                                             num_warps=warps)
        ctx.save_for_backward(xhat, weight, rstd)
        ctx.x_dtype = x.dtype
        ctx.shape, ctx.block, ctx.warps = shape, block, warps
        return y.view(*shape[:-1], cols)

    @staticmethod
    def backward(ctx, dy):
        xhat, weight, rstd = ctx.saved_tensors
        rows, cols = xhat.shape
        dy2 = dy.reshape(rows, cols).contiguous()
        dx = torch.empty((rows, cols), dtype=ctx.x_dtype, device=xhat.device)
        if rows:
            soup_rmsnorm_bwd_kernel[(rows,)](dy2, xhat, weight, rstd, dx, cols, BLOCK=ctx.block,
                                             num_warps=ctx.warps)
        return dx.view(ctx.shape), None, None, None


def rmsnorm_half(x, weight, eps, out_dtype=torch.float16):
    """weight * x / rms(x), computed in fp32, returned in `out_dtype`; no gradient for `weight`."""
    return _RMSNormHalfOut.apply(x, weight, eps, out_dtype)


def _fp16_autocast():
    return torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16


def patch_qwen2_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2RMSNorm.forward

    def forward(self, hidden_states):
        weight = self.weight
        if (hidden_states.is_cuda and not weight.requires_grad and torch.is_grad_enabled()
                and _fp16_autocast()):
            return rmsnorm_half(hidden_states, weight, self.variance_epsilon)
        return original(self, hidden_states)

    modeling_qwen2.Qwen2RMSNorm.forward = forward
