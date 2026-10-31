"""One NF4 dequantization per layer for the recompute and the backward, instead of two.

For a batch above 512 tokens bitsandbytes multiplies with `dequantize_4bit` + `F.linear`, and its
backward dequantizes the same weight again to compute the input gradient (the weights are frozen,
there is no weight gradient). With each decoder layer recomputed in the backward, that is three
dequantizations of every weight per step: the forward, the recompute, and the backward right
after it. `kDequantizeBlockwise` was ~12% of the GPU time.

Here the multiply is a small autograd function that keeps the fp16 weight it just built for its
own backward. `save_for_backward` is what makes this free of extra memory in the first forward:
the layer's checkpoint discards what is saved there, so the weights are only held in the
recompute, for the one layer being backpropagated (~94 MB for all seven projections of a layer).
Numerically it is the same two operations bitsandbytes runs above 512 tokens; smaller batches and
any call that needs no gradient go to bitsandbytes untouched.
"""

import torch

MIN_ROWS = 512  # bitsandbytes' own switch to dequantize + F.linear on this GPU


class _Linear4bitDequantOnce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, packed, bias, quant_state):
        import bitsandbytes as bnb

        weight = bnb.functional.dequantize_4bit(packed.view(-1, 1), quant_state).to(a.dtype)
        out = torch.nn.functional.linear(a, weight, bias)
        ctx.save_for_backward(weight)
        return out

    @staticmethod
    def backward(ctx, grad):
        (weight,) = ctx.saved_tensors
        return grad.to(weight.dtype) @ weight, None, None, None


def patch_bnb_matmul_4bit():
    import bitsandbytes as bnb

    original = bnb.matmul_4bit

    from .stream import streaming

    def matmul_4bit(A, B, quant_state, out=None, bias=None):
        # with streamed weights the ~94 MB of a layer's fp16 weights would sit on top of the peak
        if (not streaming() and out is None and quant_state is not None and torch.is_grad_enabled()
                and A.requires_grad and A.is_cuda and A.dtype == torch.float16
                and A.dim() >= 2 and A.numel() // A.shape[-1] > MIN_ROWS
                and len(quant_state.shape) == 2
                and not getattr(quant_state, "packing_format_for_cpu", False)
                and not (bias is not None and bias.requires_grad)):
            lead = A.shape[:-1]
            result = _Linear4bitDequantOnce.apply(A.reshape(-1, A.shape[-1]), B, bias, quant_state)
            return result.view(*lead, result.shape[-1])
        return original(A, B, quant_state=quant_state, out=out, bias=bias)

    bnb.matmul_4bit = matmul_4bit
