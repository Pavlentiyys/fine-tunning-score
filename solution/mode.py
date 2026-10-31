"""One flag shared by the fast paths: the streamed forward of a layer (stream.py) runs without
autograd. Its graph was built and thrown away (the backward recomputes the layer), so it now runs
under no_grad; our LoRA, RMSNorm, RoPE and SwiGLU functions are the same with or without autograd
and keep their fast path there, so the forward computes exactly what the recompute does (same
kernels, same dropout seeds)."""

import torch

STREAMED_FORWARD = [False]

# What the MLP of the layer being run should do with its gate/up outputs (gustash.py): "capture"
# in the streamed forward, "replay" in the recompute of the backward, None everywhere else.
GU = {"manager": None, "index": None, "mode": None}


def fast_ok():
    """Grad mode on, or inside the streamed forward of a layer."""
    return torch.is_grad_enabled() or STREAMED_FORWARD[0]
