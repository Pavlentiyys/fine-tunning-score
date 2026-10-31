# Submission: LARPERS

## Links
- Repository, final commit: https://github.com/BAITC-Hacks/hack-1b4f2831-larpers/commit/fb741edc4b98fb02990df0f059ac82da87bc1790
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link must
  be in your README):
  https://www.kaggle.com/code/mirceya/kaggle-notebook?scriptVersionId=TODO
- Demo video (up to 3 minutes): TODO
- Presentation: TODO

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| TODO | | | | |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `soup_nf4_dequant_kernel` | `solution/nf4.py` | bitsandbytes' NF4 dequantization with double-quantized scales (absmax dequant + offset add + NF4 lookup, three kernels) in the forward and backward of every 4-bit projection | 9.5 % |
| `soup_lora_dropout_kernel` | `solution/lora.py` | `nn.Dropout` in peft's LoRA and its saved bool mask: the mask is regenerated from a Philox seed in the backward | 1.9 % |
| `soup_mlp_fwd_kernel` | `solution/mlp.py` | `silu(gate) * up` (three PyTorch ops) | 1.7 % |
| `soup_mlp_bwd_kernel` | `solution/mlp.py` | SwiGLU backward, written in place over the saved gate/up | 1.4 % |
| `soup_rmsnorm_fwd_kernel` | `solution/rmsnorm.py` | Qwen2RMSNorm forward (fp32 in, fp16 out) | 1.3 % |
| `soup_rope_kernel` | `solution/rope.py` | `apply_rotary_pos_emb` (six element-wise ops), forward and inverse rotation | 1.3 % |
| `soup_ce_stream_kernel` | `solution/fused_ce.py` | lm_head logits + cross-entropy + entropy/argmax, online over streamed vocabulary chunks | 0.8 % |
| `soup_rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | RMSNorm backward from the saved fp16 normalized row | 0.5 % |
| **all** | | | **18.3 %** |

`soup_ce_rows_kernel` (`solution/fused_ce.py`) is the fused head for a GPU-resident embedding; it
does not run in the measured configuration (the embedding is kept in CPU memory).

Ideas taken from other kernels, with links (we wrote the code ourselves): chunked / fused linear
cross-entropy (Liger-Kernel FusedLinearCrossEntropy, https://github.com/linkedin/Liger-Kernel;
Cut Cross-Entropy, https://arxiv.org/abs/2411.09009), online softmax (Milakov & Gimelshein,
https://arxiv.org/abs/1805.02867), NF4 data type (QLoRA, https://arxiv.org/abs/2305.14314;
bitsandbytes' lookup table).

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
TODO: replace with the saved notebook version's run. Until then, our full run of this exact
`solution/` (harness/run_task.py, base and solution in one session):

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 86.43 | 67.94 | speedup 1.27x |
| Peak GPU memory, MiB | 14 345 | 555 | base/ours 25.8x |
| Eval loss (public rows) | 2.04965 | 2.04960 | −0.002 % vs base |

- [x] All 10 checks of `harness/check_rules.py` passed
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why, how the kernels work, results against the base (speed,
memory, quality), limits and next steps.

## Limitations
- The memory system (4-bit weights streamed from CPU memory, layer recompute, layer inputs and
  the tied embedding in CPU memory, the 152k-row head streamed in vocabulary chunks) trades speed
  for memory: without recompute and streaming the same kernels ran in ~51 s at ~3.7 GB.
- Kaggle sessions differ by up to ~20 % in absolute speed (base 86–101 s in our runs); compare
  the ratio to the base of the same session.
- Per-row attention assumes right padding (TRL's default); other masks take the padded path.
- T4 / fp16 path only; one GPU.
