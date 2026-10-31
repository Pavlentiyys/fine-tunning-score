"""NF4 dequantization with double-quantized scales in one Triton kernel.

bitsandbytes' `functional.dequantize_4bit` (called for every frozen 4-bit projection in the
backward, and by dequant.py) runs three CUDA kernels per weight for a double-quantized state:
the 8-bit absmax dequantization, `absmax += offset`, then the NF4 lookup. With 196 weights and
the Python around each call that was ~400 launches per step. Here one program reads 1024 packed
bytes (2048 weights, 32 blocks of 64) and does all of it:

    absmax[b] = code2[absmax_q[b]] * absmax2[b // 256] + offset      (fp32, as bitsandbytes)
    w[2i]     = NF4[byte_i >> 4]  * absmax[i // 32]                   (fp32 product, fp16 store)
    w[2i + 1] = NF4[byte_i & 15]  * absmax[i // 32]

The NF4 table is bitsandbytes' own (`get_4bit_type("nf4")`). The first call in a process is
checked against bitsandbytes on that weight; on any mismatch beyond rounding the patch switches
itself off and logs it, so a surprise costs speed, not correctness.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024  # packed bytes per program
WARPS = 4
_STATE = {"lut": {}, "checked": False, "off": False}


@triton.jit
def soup_nf4_dequant_kernel(packed_ptr, absq_ptr, code2_ptr, absmax2_ptr, offset_ptr, lut_ptr,
                            out_ptr, n_bytes, BLOCK: tl.constexpr, BYTES_PER_SCALE: tl.constexpr,
                            SCALES_PER_SCALE2: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = offs < n_bytes
    byte = tl.load(packed_ptr + offs, mask=live, other=0).to(tl.int32)
    scale = offs // BYTES_PER_SCALE
    q = tl.load(absq_ptr + scale, mask=live, other=0).to(tl.int32)
    absmax = tl.load(code2_ptr + q) * tl.load(absmax2_ptr + scale // SCALES_PER_SCALE2,
                                              mask=live, other=0.0) + tl.load(offset_ptr)
    hi = tl.load(lut_ptr + (byte >> 4)) * absmax
    lo = tl.load(lut_ptr + (byte & 15)) * absmax
    pair = tl.join(hi, lo).to(out_ptr.dtype.element_ty)  # [BLOCK, 2]: out[2i], out[2i + 1]
    where = out_ptr + 2 * offs[:, None] + tl.arange(0, 2)[None, :]
    tl.store(where, pair, mask=live[:, None])


def _lut(device):
    lut = _STATE["lut"].get(device)
    if lut is None:
        import bitsandbytes as bnb

        lut = bnb.functional.get_4bit_type("nf4", device=device).to(torch.float32).contiguous()
        _STATE["lut"][device] = lut
    return lut


def eligible(packed, quant_state):
    state2 = getattr(quant_state, "state2", None)
    return (not _STATE["off"] and packed.is_cuda and packed.dtype == torch.uint8
            and packed.is_contiguous() and quant_state is not None
            and quant_state.quant_type == "nf4" and quant_state.blocksize == 64
            and quant_state.dtype == torch.float16 and quant_state.nested and state2 is not None
            and state2.blocksize == 256 and quant_state.absmax.dtype == torch.uint8
            and quant_state.absmax.is_cuda and state2.absmax.dtype == torch.float32
            and state2.code.dtype == torch.float32 and torch.is_tensor(quant_state.offset)
            and quant_state.offset.is_cuda and len(quant_state.shape) == 2
            and packed.numel() * 2 == quant_state.shape[0] * quant_state.shape[1])


def nf4_dequantize(packed, quant_state):
    """fp16 weight [out, in] of a double-quantized NF4 Params4bit (see eligible())."""
    out = torch.empty(tuple(quant_state.shape), dtype=torch.float16, device=packed.device)
    n_bytes = packed.numel()
    state2 = quant_state.state2
    offset = quant_state.offset.reshape(1).to(torch.float32)
    soup_nf4_dequant_kernel[(triton.cdiv(n_bytes, BLOCK),)](
        packed, quant_state.absmax, state2.code, state2.absmax, offset, _lut(packed.device), out,
        n_bytes, BLOCK=BLOCK, BYTES_PER_SCALE=32, SCALES_PER_SCALE2=256, num_warps=WARPS)
    return out


def patch_bnb_dequantize_4bit():
    import bitsandbytes as bnb

    functional = bnb.functional
    original = functional.dequantize_4bit

    def dequantize_4bit(A, quant_state=None, absmax=None, out=None, blocksize=None,
                        quant_type="fp4"):
        transposed = torch.is_tensor(A) and A.dim() == 2 and A.shape[0] == 1
        packed = A.reshape(-1, 1) if transposed and A.is_contiguous() else A
        if (out is not None or absmax is not None or quant_state is None or packed.dim() != 2
                or packed.shape[0] == 1 or not eligible(packed, quant_state)):
            return original(A, quant_state=quant_state, absmax=absmax, out=out,
                            blocksize=blocksize, quant_type=quant_type)
        ours = nf4_dequantize(packed, quant_state)
        if transposed:  # bitsandbytes returns the transposed weight for a [1, n] packed input
            ours = ours.t()
        if not _STATE["checked"]:  # once per process: same weight through bitsandbytes
            _STATE["checked"] = True
            ref = original(A, quant_state=quant_state)  # same orientation as `ours`
            diff = (ours.float() - ref.float()).abs().max().item()
            scale = ref.float().abs().max().item()
            ok = diff <= 1e-3 * max(scale, 1e-6)
            print(f"[tl] nf4 dequant check vs bitsandbytes: max diff {diff:.3g} (max |w| "
                  f"{scale:.3g}) -> {'ours' if ok else 'OFF, bitsandbytes'}", flush=True)
            if not ok:
                _STATE["off"] = True
                return ref
        return ours

    functional.dequantize_4bit = dequantize_4bit
