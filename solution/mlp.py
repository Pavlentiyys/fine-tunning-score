"""Our own fused SwiGLU for the Qwen2 MLP: silu(gate) * up, with a backward that works in place.

Difference from a plain elementwise SwiGLU: the backward writes d_gate over the saved `gate`
buffer and d_up over `up` instead of allocating two more [rows, 8960] tensors. For batch 4
x 512 tokens that is 2 x 37 MB per layer that never exist next to the saved activations, which
lowers the peak right where the backward starts.

`down_proj` is a frozen 4-bit layer: bitsandbytes' MatMul4Bit saves only the weight, never its
input, so the product is not kept for backward and `gate`/`up` are the only tensors that live
from the forward to the backward of the layer.
"""

import torch
import triton
import triton.language as tl

from .mode import fast_ok

COLS = 1024


@triton.jit
def soup_mlp_fwd_kernel(gate_ptr, up_ptr, out_ptr, n_cols, stride, COLS: tl.constexpr):
    """One program per row, looping over the row in COLS-wide tiles."""
    base = tl.program_id(0).to(tl.int64) * stride
    lane = tl.arange(0, COLS)
    for start in range(0, n_cols, COLS):
        cols = start + lane
        live = cols < n_cols
        g = tl.load(gate_ptr + base + cols, mask=live, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + base + cols, mask=live, other=0.0).to(tl.float32)
        h = (g * u) / (1.0 + tl.exp2(-1.4426950408889634 * g))
        tl.store(out_ptr + base + cols, h.to(tl.float16), mask=live)


@triton.jit
def soup_mlp_bwd_kernel(dout_ptr, gate_ptr, up_ptr, n_cols, stride, COLS: tl.constexpr):
    """d_gate is stored over gate, d_up over up (both read before they are written)."""
    base = tl.program_id(0).to(tl.int64) * stride
    lane = tl.arange(0, COLS)
    for start in range(0, n_cols, COLS):
        cols = start + lane
        live = cols < n_cols
        g = tl.load(gate_ptr + base + cols, mask=live, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + base + cols, mask=live, other=0.0).to(tl.float32)
        d = tl.load(dout_ptr + base + cols, mask=live, other=0.0).to(tl.float32)
        sig = tl.sigmoid(g)
        # d/dg [g * sig(g)] = sig * (1 + g * (1 - sig))
        tl.store(up_ptr + base + cols, (d * g * sig).to(tl.float16), mask=live)
        tl.store(gate_ptr + base + cols, (d * u * sig * (1.0 + g * (1.0 - sig))).to(tl.float16),
                 mask=live)


class _SwiGLUInplace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        shape = gate.shape
        gate2 = gate.reshape(-1, shape[-1]).contiguous()
        up2 = up.to(gate2.dtype).reshape(-1, shape[-1]).contiguous()
        rows, cols = gate2.shape
        out = torch.empty_like(gate2)
        grid = (rows,)
        soup_mlp_fwd_kernel[grid](gate2, up2, out, cols, gate2.stride(0), COLS=COLS)
        ctx.save_for_backward(gate2, up2)
        ctx.shape = shape
        return out.view(shape)

    @staticmethod
    def backward(ctx, dout):
        gate2, up2 = ctx.saved_tensors
        rows, cols = gate2.shape
        dout2 = dout.reshape(rows, cols).to(gate2.dtype).contiguous()
        grid = (rows,)
        soup_mlp_bwd_kernel[grid](dout2, gate2, up2, cols, gate2.stride(0), COLS=COLS)
        return gate2.view(ctx.shape), up2.view(ctx.shape)


def swiglu_inplace(gate, up):
    return _SwiGLUInplace.apply(gate, up)


def patch_qwen2_mlp_own():
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2MLP.forward

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if act != "silu" or not x.is_cuda or not fast_ok():
            return original(self, x)
        return self.down_proj(swiglu_inplace(self.gate_proj(x), self.up_proj(x)))

    modeling_qwen2.Qwen2MLP.forward = forward
