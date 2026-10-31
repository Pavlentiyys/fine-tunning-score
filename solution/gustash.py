"""Skip the MLP's three GEMMs in the recompute of the backward.

Every decoder layer runs three times per step: the streamed forward, the recompute in the backward
(which rebuilds the autograd graph), and the backward itself. In the recompute the MLP is 88 % of
the layer's FLOPs (gate_proj, up_proj and down_proj, 3 x 13.8 M of 46.8 M weights), and its result
is not even read: the layer output of the recompute only gives `torch.autograd.grad` a graph to walk.

So the streamed forward copies the MLP's `gate` and `up` outputs to pinned CPU memory
(`LayerManager.gu_stash`), the backward copies them back one layer ahead, and the MLP of the
recompute becomes `_MlpReplay`: its forward returns zeros without computing anything, its backward
does what the graph of `down_proj(swiglu(gate_proj(x), up_proj(x)))` would do, from the saved gate
and up (the same values the forward produced, bit for bit):

    d_act = d_out @ W_down;   d_gate, d_up = swiglu_bwd(d_act, gate, up);   dx = d_gate @ W_gate + d_up @ W_up

with the NF4 weights dequantized by our kernel when they are needed (the weights are frozen, no
weight gradients). The attention half of the layer is still recomputed: the norm after it saves
tensors that depend on its result.

Used only when `stream.GU_STASH` is on and the PCIe bus test at the first step passes; any layer
without a matching stash is recomputed the normal way.
"""

import torch

from .mlp import COLS, soup_mlp_bwd_kernel
from .mode import GU
from .nf4 import eligible, nf4_dequantize


def _packed(linear):
    return linear.weight.t().reshape(-1, 1)  # what bitsandbytes' matmul_4bit gets as `B`


def _plain_linears(mlp):
    """(gate, up, down) when all three are frozen, bias-free, NF4 layers our kernel dequantizes."""
    out = []
    for linear in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        state = getattr(getattr(linear, "weight", None), "quant_state", None)
        if (linear.__class__.__name__ != "Linear4bit" or getattr(linear, "bias", None) is not None
                or state is None or not eligible(_packed(linear), state)):
            return None
        out.append(linear)
    return out


def usable_for_capture(mlp, gate, up):
    return (gate.dtype == torch.float16 and up.dtype == torch.float16 and gate.dim() == 2
            and gate.shape == up.shape and gate.is_contiguous() and up.is_contiguous()
            and _plain_linears(mlp) is not None)


def capture(mlp, gate, up):
    if usable_for_capture(mlp, gate, up):
        GU["manager"].gu_stash(GU["index"], gate, up)


def replay(mlp, x):
    """The MLP output's stand-in for the recompute, or None to compute the MLP the normal way."""
    linears = _plain_linears(mlp)
    if linears is None or x.dim() != 2 or x.dtype != torch.float16:
        GU["manager"].gu_drop(GU["index"])  # a stash this layer cannot use must not stay alive
        return None
    taken = GU["manager"].gu_take(GU["index"], x.shape[0])
    if taken is None:
        return None
    gate, up = taken
    gate_l, up_l, down_l = linears
    nxt = (GU["manager"], GU["index"] - 1) if PREFETCH_AFTER_MLP else None
    return _MlpReplay.apply(x, gate, up, _packed(gate_l), _packed(up_l), _packed(down_l),
                            gate_l.weight.quant_state, up_l.weight.quant_state,
                            down_l.weight.quant_state, nxt)


# The next layer's gate/up are copied back once this layer's are freed (end of the MLP backward),
# not at the start of the layer's backward: both layers' [tokens, 8960] pairs were on the GPU at the
# peak. The copy still overlaps the attention half of this layer's backward.
PREFETCH_AFTER_MLP = True


class _MlpReplay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, up, packed_gate, packed_up, packed_down, state_gate, state_up,
                state_down, nxt=None):
        ctx.save_for_backward(packed_gate, packed_up, packed_down)
        ctx.gu = [gate, up]  # held here, not saved: the backward drops them as soon as it is done
        ctx.states, ctx.nxt = (state_gate, state_up, state_down), nxt
        return x.new_zeros(x.shape[0], state_down.shape[0])  # never read: only the graph is used

    @staticmethod
    def backward(ctx, d_out):
        packed_gate, packed_up, packed_down = ctx.saved_tensors
        gate, up = ctx.gu
        ctx.gu = None
        state_gate, state_up, state_down = ctx.states
        weight = nf4_dequantize(packed_down, state_down)  # [hidden, inter]
        d_act = (d_out.to(weight.dtype) @ weight).contiguous()  # [tokens, inter]
        del weight
        rows, cols = gate.shape
        soup_mlp_bwd_kernel[(rows,)](d_act, gate, up, cols, gate.stride(0), COLS=COLS)
        del d_act  # gate now holds d_gate and up holds d_up
        weight = nf4_dequantize(packed_gate, state_gate)  # [inter, hidden]
        d_x = gate @ weight
        del weight
        weight = nf4_dequantize(packed_up, state_up)
        d_x = torch.addmm(d_x, up, weight)
        del weight, gate, up  # freed in stream order: the queued GEMMs still read them first
        if ctx.nxt is not None:
            manager, index = ctx.nxt
            ctx.nxt = None
            manager.gu_prefetch(index)
        return (d_x, None, None, None, None, None, None, None, None, None)
