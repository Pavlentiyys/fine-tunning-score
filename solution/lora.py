"""LoRA on a 4-bit layer with the dropout mask regenerated instead of stored.

peft's `lora.Linear4bit.forward` (q_proj and v_proj in every layer) clones the base output, runs
`nn.Dropout` on the input, which saves its bool mask, and `lora_A` saves the dropped-out input for
its weight gradient: two [tokens, 1536] tensors per adapter kept to the backward, four per layer.

Here the mask comes from Triton's counter-based Philox generator (`tl.rand(seed, offset)`): the
backward regenerates exactly the same mask from the same seed, so the only tensors kept are the
input x (one tensor shared by q_proj's and v_proj's adapters) and h = dropout(x) @ A^T, which is
[tokens, r=16]. The B product is added to the base output in one `addmm`, without the clone.

Same LoRA (same weights, scaling, dropout p and fp16 compute under autocast); the dropout masks are
drawn from another generator than torch's, i.e. different random draws of the same distribution.
"""

import weakref

import torch
import triton
import triton.language as tl

from .mode import fast_ok

BLOCK = 1024


@triton.jit(do_not_specialize=["seed", "n_elements"])
def soup_lora_dropout_kernel(x_ptr, out_ptr, n_elements, seed, p, scale, BLOCK: tl.constexpr):
    """out = x * keep / (1 - p), keep = philox(seed, flat index) >= p; the same (seed, index) gives
    the same mask in forward (on x) and backward (on the gradient)."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = offs < n_elements
    keep = tl.rand(seed, offs.to(tl.int32)) >= p  # < 2^31 elements per tensor
    x = tl.load(x_ptr + offs, mask=live, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, tl.where(keep, x * scale, 0.0).to(out_ptr.dtype.element_ty),
             mask=live)


def _dropout(x, seed, p):
    out = torch.empty_like(x)
    n = x.numel()
    soup_lora_dropout_kernel[(triton.cdiv(n, BLOCK),)](x, out, n, seed, p, 1.0 / (1.0 - p),
                                                       BLOCK=BLOCK)
    return out


_HALF = {}  # id(weight) -> (version, data_ptr, fp16 copy): one cast per optimizer step, not six


def _half(weight):
    """`weight.to(float16)`, cached until the weight changes (the optimizer updates it in place,
    which bumps its version counter): forward, recompute and backward of a step share one cast.
    The entry also holds a weak reference, so a new tensor that reuses a freed one's id never
    gets its copy."""
    key = id(weight)
    version, ptr = weight._version, weight.data_ptr()
    cached = _HALF.get(key)
    if cached is not None and cached[0]() is weight and cached[1] == version             and cached[2] == ptr:
        return cached[3]
    half = weight.detach().to(torch.float16)
    _HALF[key] = (weakref.ref(weight), version, ptr, half)
    return half


class _LoraAdd(torch.autograd.Function):
    """base + scaling * dropout(x) @ A^T @ B^T, with x [M, K] fp16 and base [M, N] fp16."""

    @staticmethod
    def forward(ctx, x, base, weight_a, weight_b, p, scaling, seed):
        a16, b16 = _half(weight_a), _half(weight_b)
        h = _dropout(x, seed, p) @ a16.t()
        out = torch.addmm(base, h, b16.t(), alpha=scaling)
        ctx.save_for_backward(x, h, weight_a, weight_b)
        ctx.p, ctx.scaling, ctx.seed = p, scaling, seed
        return out

    @staticmethod
    def backward(ctx, grad):
        x, h, weight_a, weight_b = ctx.saved_tensors
        p, scaling, seed = ctx.p, ctx.scaling, ctx.seed
        a16, b16 = _half(weight_a), _half(weight_b)
        grad = grad.contiguous()
        grad_b = (grad.t() @ h).float().mul_(scaling)
        dh = (grad @ b16).mul_(scaling)
        grad_a = (dh.t() @ _dropout(x, seed, p)).float()
        grad_x = _dropout(dh @ a16, seed, p)
        return grad_x, grad, grad_a, grad_b, None, None, None


_GEN = []


def _next_seed():
    """Seeds from a private generator seeded by the run's seed, so torch's own CPU and CUDA random
    streams (data order, anything else) are not consumed."""
    if not _GEN:
        gen = torch.Generator()
        gen.manual_seed(torch.initial_seed() ^ 0x5EED)
        _GEN.append(gen)
    return int(torch.randint(0, 2**31 - 1, (1,), generator=_GEN[0]).item())


class _Replay:
    seeds = None  # seeds handed out in order inside a recomputed region (see replay_seeds)


def draw_seeds(count):
    return [_next_seed() for _ in range(count)]


class replay_seeds:
    """Inside this block the LoRA layers take their dropout seeds from `seeds`, in call order. A
    region that runs twice (forward, then recomputed in backward) gets the same list both times,
    so the recomputed dropout masks are the ones the forward used."""

    def __init__(self, seeds):
        self.seeds, self.saved = list(seeds), None

    def __enter__(self):
        self.saved, _Replay.seeds = _Replay.seeds, list(self.seeds)

    def __exit__(self, *exc):
        _Replay.seeds = self.saved
        return False


def _take_seed():
    if _Replay.seeds:
        return _Replay.seeds.pop(0)
    return _next_seed()


_STATIC = {}  # id(layer) -> (weight_a, weight_b, p, scaling) or None: the adapter set is fixed


def _fast_path(layer, x, args, kwargs):
    if (args or kwargs or layer.disable_adapters or layer.merged or not layer.training
            or not fast_ok() or not x.is_cuda or x.dtype != torch.float16
            or not torch.is_autocast_enabled("cuda")
            or torch.get_autocast_dtype("cuda") != torch.float16):
        return None
    key = (id(layer), tuple(layer.active_adapters))
    if key not in _STATIC:
        _STATIC[key] = _static_path(layer)
    return _STATIC[key]


def _static_path(layer):
    adapters = [name for name in layer.active_adapters if name in layer.lora_A.keys()]
    if len(adapters) != 1 or adapters[0] in getattr(layer, "lora_variant", {}):
        return None
    name = adapters[0]
    dropout = layer.lora_dropout[name]
    if not isinstance(dropout, torch.nn.Dropout) or not 0.0 < dropout.p < 1.0:
        return None
    a, b = layer.lora_A[name], layer.lora_B[name]
    if a.bias is not None or b.bias is not None:
        return None
    return a.weight, b.weight, float(dropout.p), float(layer.scaling[name])


def patch_peft_lora_4bit():
    from peft.tuners.lora import bnb as lora_bnb

    cls = getattr(lora_bnb, "Linear4bit", None)
    if cls is None:
        return
    original = cls.forward

    def forward(self, x, *args, **kwargs):
        found = _fast_path(self, x, args, kwargs)
        if found is None:
            return original(self, x, *args, **kwargs)
        weight_a, weight_b, p, scaling = found
        base = self.base_layer(x)
        shape = base.shape
        out = _LoraAdd.apply(x.reshape(-1, x.shape[-1]).contiguous(),
                             base.reshape(-1, shape[-1]), weight_a, weight_b, p, scaling,
                             _take_seed())
        return out.view(shape)

    cls.forward = forward
