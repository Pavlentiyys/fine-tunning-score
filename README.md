# Soup Kernel Challenge — team LARPERS

Our `solution/` makes the fixed `soup train` run (Qwen2.5-1.5B-Instruct, NF4, LoRA r=16 on
q/v, batch 4, 64 steps, Kaggle T4) fit in **about 0.5 GB of GPU memory instead of 14.3 GB**, with
the same adapter quality and a shorter run. Eight own Triton kernels do the arithmetic. Around them
we built a memory system: layer weights streamed from CPU memory, layer recomputation, no padding,
and the 152k-token head streamed in vocabulary chunks.

- Task statement and the original starter kit: https://github.com/MakazhanAlpamys/soup-kernel-challenge
- Final commit: **TODO (filled in at submission)**
- Public Kaggle notebook (saved version, `COMMIT` = final commit): **TODO**

## Results

From `harness/run_task.py` on a Kaggle T4, base Soup and our solution in the same session. The
table is from our last full run. **TODO: replace it with the final `--repeats 2` run before submission.**

| | Base Soup 0.77.0 | Our solution | Ratio |
|---|---|---|---|
| Whole `soup train` command | 75.5 s | 69.9 s | speedup 1.08x |
| Peak GPU memory (nvidia-smi) | 14 345 MiB | 519 MiB | memory base/solution **27.6x** |
| Eval loss, public rows 40000–40255 | 2.04969 | 2.04956 | −0.006 % |

All 10 checks of `harness/check_rules.py` pass. Our kernels take 11.6 % of the training time
(check 4, steps 3+, CUDA events). The loss curve follows the base within 0.05 % on average; the
first step differs by 0.02 %.

How we got there (each row is one measured run on the same task; the time ratio is solution/base
within that run, because Kaggle sessions differ by up to ~20 % in absolute speed):

| Step | Peak, MiB | Time solution/base |
|---|---|---|
| Base Soup | 14 345 | 1.00 |
| Fused head + CE, RMSNorm with fp16 output | 6 463 | 0.82 |
| + no padding outside attention, fp16 embedding, RoPE, own SwiGLU | 4 641 | 0.72 |
| + attention projections without padding, LoRA kernel | 4 289 | 0.71 |
| + layer recomputation, smaller head chunks | 1 721 – 2 135 | 0.83 |
| + 4-bit layer weights streamed from pinned CPU memory | 1 371 | 0.84 |
| + no allocator warm-up, embedding/head kept on CPU and streamed | 593 | 0.88 |
| + per-row causal attention without padding | 519 | 0.93 |

## How to run

On Kaggle (this is how the jury runs it):

1. Create → New Notebook → File → Import Notebook → `templates/kaggle_notebook.ipynb`.
2. Settings: Accelerator **GPU T4 x2**, Internet **on**. For a private repository, add a Kaggle
   secret `GITHUB_TOKEN` (a read-only token).
3. In the first code cell set `REPO_URL` to this repository and `COMMIT` to the final commit.
4. **Run All**, about 12 minutes. The notebook installs the pinned packages, prepares the data,
   runs our kernel tests (`pytest solution`), trains base Soup and our solution, scores both
   adapters and prints the checks.

Locally on Linux with a T4 (the numbers only count on Kaggle):

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q          # our kernels against plain PyTorch
python harness/run_task.py            # add --repeats 2 for the jury's order
python harness/check_rules.py
```

Restart the notebook kernel before a measured run if you ran any CUDA code in the notebook
process itself. The harness sums the memory of every process on the GPU.

## What we did and why

Base Soup on this task peaks at 14.3 GB, for three reasons:
- the logits over 151 936 tokens are kept in fp16 and fp32, plus their copies for TRL's entropy
  and accuracy and for Soup's rewind log;
- `prepare_model_for_kbit_training` keeps the residual stream, the norms and the tied
  embedding/head in fp32, so every activation of every layer is stored in fp32;
- all 28 layers' activations live until the backward.

The time goes to element-wise ops and casts (44 %) and to matmuls on padded batches.

### Own Triton kernels (`@triton.jit`, all in `solution/`)

| Kernel | File | What it does |
|---|---|---|
| `soup_ce_stream_kernel` | `fused_ce.py` | One vocabulary chunk of the output head: an online-softmax update per row (max, sum of exp, sum of exp·x for the entropy, argmax, target logit). For labelled rows it also turns the logits into exp(x − max) for the gradient. |
| `soup_ce_rows_kernel` | `fused_ce.py` | The same head for a GPU-resident weight: LSE, entropy, argmax and softmax − onehot in place, per row. Used when `EMBEDDING_ON_CPU = False`, so it is not launched in the final configuration. |
| `soup_rmsnorm_fwd_kernel` / `_bwd_kernel` | `rmsnorm.py` | RMSNorm. Reads the fp32 residual and writes fp16, the dtype the next matmul uses anyway. Saves the normalized row in fp16 plus one rstd per row. The backward recomputes from these (frozen weight: input gradient only). |
| `soup_mlp_fwd_kernel` / `_bwd_kernel` | `mlp.py` | SwiGLU `silu(gate)·up`. The backward writes d_gate/d_up over the saved gate/up buffers instead of allocating new ones. Our own design, not the example's: similarity to the example is 0.58 by `check_rules.py`. |
| `soup_rope_kernel` | `rope.py` | Rotary embedding of q and k in one launch per tensor, fp32 math, fp16 out. The backward is the inverse rotation from the same kernel, and nothing is saved. Replaces six element-wise ops with an fp32 result plus SDPA's fp16 copies. |
| `soup_lora_dropout_kernel` | `lora.py` | LoRA dropout from Triton's counter-based Philox (`tl.rand(seed, offset)`). The backward regenerates the same mask from the seed instead of storing the mask and the dropped-out input. |

### Fused output head and loss (`fused_ce.py`)
- The head is never materialized as [batch, seq, 151 936]. In the final configuration its weight
  sits in pinned CPU memory and is streamed in chunks of 4096 vocabulary rows. While one chunk
  computes, the next one is copied on a side stream into a persistent double buffer.
- **One pass over the vocabulary is enough.** Per row we keep the online-softmax state across
  chunks and accumulate `acc = Σ exp(x − m)·W_chunk`, rescaled by `exp(m_old − m_new)` when the max
  grows. Then softmax·W = acc / Σexp, and the gradient of the hidden state is softmax·W − W[target].
  So each weight byte crosses PCIe once per step.
- Only rows that need logits are computed: labelled rows for the loss, accuracy and rewind, and
  attended rows for TRL's entropy. Padding rows are skipped.
- **TRL's `entropy`, `mean_token_accuracy`, `num_tokens` and Soup's per-row rewind losses** are
  logged by the same formulas from the kernel's outputs. Soup reads the per-row losses through its
  own "step without logits" protocol (`_soup_chunked_step.token_loss`). Soup's built-in
  `stream_loss_chunk` mode stays off.

### No padding (`unpad.py`)
- TRL pads each batch to its longest row. With rows of ~220 tokens on average against a longest of
  ~350–512, about 40 % of the positions are padding.
- The residual stream is kept as [real tokens, hidden]. The norms, residual adds, q/k/v/o
  projections with LoRA and the MLP all run on real tokens only.
- **Attention runs per row, causally, on the row's real tokens.** That is exactly what the causal +
  padding mask allowed for every real token. The work is Σ L² instead of batch · L_max², and no
  padded tensor is built.
  - RoPE tables are gathered once at each real token's position.
  - K/V heads are repeated as in transformers' `repeat_kv`.
  - SDPA runs without a mask and without `enable_gqa`, i.e. the memory-efficient causal kernel
    (on a T4, GQA would fall back to the math kernel).
- No rows are mixed across batches, as the rules require.

### Memory system
- **Layer recomputation** (`unpad.py`, `stream.py`): only each decoder layer's input is kept for
  the backward, and the layer is recomputed there. This is our own autograd function; Soup's
  `gradient_checkpointing` stays off. LoRA dropout seeds are drawn before the layer and replayed,
  so the recomputed masks equal the forward's.
- **4-bit weights streamed from CPU** (`stream.py`).
  - Right after bitsandbytes quantizes a weight (`Params4bit._quantize`), its packed data and
    scales go to CPU memory, and the GPU keeps empty placeholders. So the model is never on the GPU
    as a whole, not even while loading.
  - On the first step the layers are packed into one pinned arena.
  - A layer is fetched with one host-to-device copy into a ring of persistent GPU buffers, one
    layer ahead, on a side stream synchronized by events. There are no side-stream allocations and
    no `record_stream`: both had made the allocator reserve about twice what was alive.
- **Layer inputs offloaded**: the fp32 input of each layer is copied to pinned memory
  asynchronously. In the backward it is brought back one layer ahead.
- **Tied embedding/head on the CPU** (`prepare.py`).
  - transformers materializes this one tensor in CPU memory instead of the GPU.
  - The embedding keeps fp16 through `prepare_model_for_kbit_training`: the checkpoint is bf16,
    which fp16 represents exactly in the range used, and the head already runs in fp16 under
    autocast.
  - It is pinned against `module.to(device)` (e.g. inside `accelerate.prepare_model`).
  - The forward looks up only the real tokens' rows on the CPU.
- **No allocator warm-up**: transformers' `caching_allocator_warmup` reserves the model's full
  size in one allocation before loading. With weights leaving the GPU as they are quantized, that
  reservation (~1.07 GB) was the peak of the whole process.
- **fp16 activations**: with the norms writing fp16, q/k/v/gate/up and the SwiGLU product are fp16
  instead of fp32. The matmuls already ran in fp16, so their inputs are unchanged.

### Time
- Our Triton kernels are compiled in a background thread while Soup loads the model
  (`warmup.py`, `JITFunction.warmup`, nothing launched). Every run starts with an empty cache, and
  the compilation counts in the measured time. No compiled cache is shipped.
- When weights are not streamed, `dequant.py` dequantizes each NF4 weight once for the recompute
  and the backward instead of twice (`kDequantizeBlockwise` was ~12 % of GPU time). It is off with
  streaming, where those ~94 MB would add to the peak.

### Switches (all in `solution/`, for experiments; the defaults are the submitted configuration)
`stream.ENABLED`, `stream.OFFLOAD_INPUTS`, `stream.PREFETCH_DEPTH`, `unpad.RECOMPUTE_LAYERS`,
`unpad.PER_ROW_ATTENTION`, `prepare.EMBEDDING_ON_CPU`, `prepare.NO_ALLOCATOR_WARMUP`,
`fused_ce.HEAD_CHUNK`.

## Ideas and sources (all code is ours)
- Fused / chunked linear cross-entropy: Liger-Kernel `FusedLinearCrossEntropy`
  (https://github.com/linkedin/Liger-Kernel) and Apple's Cut Cross-Entropy (Wijmans et al., 2024,
  https://arxiv.org/abs/2411.09009).
- Online softmax with rescaling: Milakov & Gimelshein, 2018 (https://arxiv.org/abs/1805.02867),
  as used in FlashAttention (Dao et al., https://arxiv.org/abs/2205.14135).
- Seeded dropout without storing the mask: Triton tutorial "Low-Memory Dropout"
  (https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html).
- RMSNorm / LayerNorm kernels: Triton tutorial "Layer Normalization"
  (https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html).
- Variable-length attention without padding: FlashAttention's varlen interface, done here as
  per-row SDPA calls.
- Activation recomputation: Chen et al., 2016 (https://arxiv.org/abs/1604.06174).
  CPU offloading of weights and activations: ZeRO-Offload (Ren et al., 2021,
  https://arxiv.org/abs/2101.06840).
- Patching a model through one `apply()` call: the way Unsloth plugs into training loops.
- NF4 / QLoRA: Dettmers et al., 2023 (https://arxiv.org/abs/2305.14314).

## Limitations and what did not work
- **Memory first, time second.** Recomputation (one more forward per layer) and PCIe streaming
  cost time. Without them, the same kernels run the task in ~0.71x of the base time but at ~4.3 GB
  (commit `073c6a0`). We chose the memory end of that trade-off.
- The first version of weight streaming (one pinned tensor and one copy per weight) cost ~12 s.
  Packing into one arena and copying one layer at a time fixed most of it.
- Our first per-row attention was slower than the padded one. Per-row RoPE and K/V copies were
  too many small launches for Kaggle's CPU. RoPE and the K/V repeat now run once for all tokens.
- `nvidia-smi` samples every 100 ms. Short allocator spikes (an fp32/fp16 embedding pair for a
  few ms) made check 9 flaky until we removed the spikes themselves.
- The LoRA dropout masks come from Philox, not from torch's generator: the same distribution,
  different draws. The loss curve stays within 0.05 % of the base on average.
- Shapes specific to Qwen2.5-1.5B appear in two places: the embedding shape in `prepare.py` and the
  warm-up shapes in `warmup.py`. Another model would still work, without those two optimizations.
- `train.log` gets diagnostic lines (`[mem]`, `[time]`, `[load]`, `[warmup]`). They only read
  counters.
- Tested on a Kaggle T4 only. On Ampere and newer GPUs Soup trains in bf16, a different path.

## Repository layout
| Path | What |
|---|---|
| `solution/__init__.py` | `apply()`: installs everything below |
| `solution/fused_ce.py` | fused head + loss + metrics, CPU-streamed head |
| `solution/rmsnorm.py`, `mlp.py`, `rope.py`, `lora.py` | kernels and their autograd functions |
| `solution/unpad.py` | the decoder without padding, per-row attention, recomputation |
| `solution/stream.py` | 4-bit weight and layer-input streaming |
| `solution/prepare.py` | loading: embedding on CPU, no allocator warm-up, fp16 embedding |
| `solution/dequant.py`, `warmup.py`, `timeline.py` | dequantize once, background compile, timestamps |
| `solution/test_kernels.py` | our kernels against plain PyTorch (`pytest solution`) |
| `harness/`, `task/`, `tests/`, `templates/` | the starter kit, unchanged |
