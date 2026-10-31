"""Compile our Triton kernels in a background thread while Soup loads the model.

Every measured run starts with an empty Triton cache, and the compilation is part of the measured
time: in the log the first training step took ~5 s against ~0.6 s for the next ones. `apply()`
runs before Soup reads the data and loads and quantizes the model (several seconds of mostly I/O
and GPU work), so the compilation can happen in that time instead of inside the first step.

`JITFunction.warmup` compiles without launching anything. The arguments mirror the training calls
(Qwen2.5-1.5B shapes, dtypes, constexprs, num_warps), because Triton specializes on argument
dtypes, 16-byte alignment and integers divisible by 16: a mismatch only means the kernel compiles
again on first use, as it would without this file. This compiles in the run itself; no cache is
shipped or reused across runs.
"""

import threading

import torch

HIDDEN, INTER, VOCAB, HEAD_DIM = 1536, 8960, 151936, 128
Q_HEADS, KV_HEADS = 12, 2


def _compile_all():
    from . import fused_ce, lora, mlp, rmsnorm, rope

    def make(dtype, *shape):
        return torch.empty(shape or (16,), device="cuda", dtype=dtype)

    f16, f32, i32 = torch.float16, torch.float32, torch.int32
    grid = (1,)

    block = 2048  # next power of two of HIDDEN, as rmsnorm.py picks it
    rmsnorm.soup_rmsnorm_fwd_kernel.warmup(
        make(f32), make(f32), make(f16), make(f16), make(f32), HIDDEN, 1e-6,
        BLOCK=block, num_warps=8, grid=grid)
    rmsnorm.soup_rmsnorm_bwd_kernel.warmup(
        make(f16), make(f16), make(f32), make(f32), make(f32), HIDDEN,
        BLOCK=block, num_warps=8, grid=grid)

    mlp.soup_mlp_fwd_kernel.warmup(make(f16), make(f16), make(f16), INTER, INTER,
                                   COLS=mlp.COLS, grid=grid)
    mlp.soup_mlp_bwd_kernel.warmup(make(f16), make(f16), make(f16), INTER, INTER,
                                   COLS=mlp.COLS, grid=grid)

    # strides of a [B, T, H, D] projection viewed as [B, H, T, D]: all multiples of 128
    for heads in (Q_HEADS, KV_HEADS):
        for inverse in (False, True):
            rope.soup_rope_kernel.warmup(
                make(f16), make(f16), make(f32), make(f32),
                512 * heads * HEAD_DIM, HEAD_DIM, heads * HEAD_DIM,
                512 * heads * HEAD_DIM, HEAD_DIM, heads * HEAD_DIM,
                0, HEAD_DIM, 512, heads,
                HALF=HEAD_DIM // 2, HEADS=rope.ROPE_HEADS, INVERSE=inverse, num_warps=4,
                grid=grid)

    lora.soup_lora_dropout_kernel.warmup(make(f16), make(f16), 1000, 1234, 0.05, 1.0 / 0.95,
                                         BLOCK=lora.BLOCK, grid=grid)

    fused_ce.soup_ce_stream_kernel.warmup(
        make(f16), fused_ce.HEAD_CHUNK, fused_ce.HEAD_CHUNK, 0, make(i32), make(f32), make(f32),
        make(f32), make(f32), make(i32), make(f32), make(f32), 100, BLOCK_V=fused_ce.BLOCK_V,
        num_warps=8, grid=grid)

    from . import nf4

    nf4.soup_nf4_dequant_kernel.warmup(
        make(torch.uint8), make(torch.uint8), make(f32, 256), make(f32), make(f32), make(f32),
        make(f16, 32), 1536, BLOCK=nf4.BLOCK, BYTES_PER_SCALE=32, SCALES_PER_SCALE2=256,
        num_warps=nf4.WARPS, grid=grid)

    from .prepare import EMBEDDING_ON_CPU

    # the GPU-resident head kernel only runs when the embedding is not kept on the CPU
    for write_grad in (() if EMBEDDING_ON_CPU else (False, True)):
        fused_ce.soup_ce_rows_kernel.warmup(
            make(f16), VOCAB, make(i32), make(f32), make(f32), make(i32), VOCAB,
            WRITE_GRAD=write_grad, BLOCK_V=fused_ce.BLOCK_V, num_warps=8, grid=grid)


def _safe_compile():
    import time

    started = time.perf_counter()
    try:
        _compile_all()
    except Exception as exc:  # noqa: BLE001 - a failed warm-up just means compiling on first use
        print(f"[warmup] skipped: {exc!r}", flush=True)
        return
    print(f"[warmup] Triton kernels compiled in {time.perf_counter() - started:.1f} s", flush=True)


_THREAD = []


def start_background_compile():
    if not torch.cuda.is_available():
        return
    # not a daemon: the interpreter waits for it at exit instead of killing it mid-compile
    thread = threading.Thread(target=_safe_compile, name="soup-triton-warmup", daemon=False)
    thread.start()
    _THREAD.append(thread)
    _join_at_first_step()


def _join_at_first_step():
    """Join the compile thread when the first training step starts, before our first kernel
    launch, so the training never compiles a kernel the thread is still compiling."""
    try:
        import transformers
    except Exception:  # noqa: BLE001 - no transformers (fake modules in tests): nothing to hook
        return
    original = transformers.Trainer.training_step

    def training_step(self, *args, **kwargs):
        while _THREAD:
            _THREAD.pop().join(900)
        return original(self, *args, **kwargs)

    transformers.Trainer.training_step = training_step
