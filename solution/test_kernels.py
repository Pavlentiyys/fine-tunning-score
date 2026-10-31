"""Correctness of your kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`. Add a test like this for every kernel you write.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


@pytest.mark.parametrize("in_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(4, 300, 1536), (2, 7, 1000)])
def test_rmsnorm_half_matches_pytorch(in_dtype, shape):
    from solution.rmsnorm import rmsnorm_half

    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=in_dtype, requires_grad=True)
    weight = torch.rand(shape[-1], device="cuda", dtype=torch.float32) + 0.5
    grad = torch.randn(shape, device="cuda", dtype=torch.float16)

    xf = x.float()
    ref = weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6))
    ref.backward(grad.float())
    ref_dx = x.grad.float()
    x.grad = None

    out = rmsnorm_half(x, weight, 1e-6)
    assert out.dtype == torch.float16
    out.backward(grad)
    torch.testing.assert_close(out.float(), ref, rtol=2e-3, atol=2e-3)
    assert x.grad.dtype == in_dtype
    torch.testing.assert_close(x.grad.float(), ref_dx, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("on_cpu", [False, True])
@pytest.mark.parametrize("vocab", [151936, 5003])
def test_fused_head_loss_matches_pytorch(vocab, on_cpu):
    from solution.fused_ce import fused_head_loss, fused_head_loss_cpu_weight

    torch.manual_seed(0)
    batch, seq, width = 3, 70, 256
    hidden = (torch.randn(batch, seq, width, device="cuda") * 0.5).requires_grad_(True)
    weight = torch.randn(vocab, width, device="cuda") * 0.05
    weight16 = weight.half()
    labels = torch.randint(0, vocab, (batch, seq), device="cuda")
    labels[:, :20] = -100  # prompt
    attention_mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
    attention_mask[1, 50:] = 0  # padding at the end of row 1
    labels[1, 50:] = -100
    num_items = (labels[:, 1:] != -100).sum()

    # reference: what base Soup computes from fp16 logits (fp32 loss, TRL entropy and accuracy,
    # Soup's per-row token losses)
    logits = (hidden.half() @ weight16.t())
    shift = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:]
    ref_loss = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, vocab), shift.reshape(-1), ignore_index=-100,
        reduction="sum") / num_items
    ref_loss.backward()
    ref_grad = hidden.grad.clone()
    hidden.grad = None
    with torch.no_grad():
        logp = torch.log_softmax(logits.float(), -1)
        ref_entropy = -(logp.exp() * logp).sum(-1)
        ref_token = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, vocab), shift.reshape(-1), ignore_index=-100,
            reduction="none").view(batch, seq)
        mask = shift != -100
        ref_correct = ((logits.argmax(-1) == shift) & mask).sum()

    if on_cpu:
        loss, step = fused_head_loss_cpu_weight(hidden, weight16.cpu().pin_memory(), labels,
                                                attention_mask, num_items)
    else:
        loss, step = fused_head_loss(hidden, weight16, labels, attention_mask, num_items)
    loss.backward()
    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(hidden.grad, ref_grad, rtol=2e-2, atol=2e-5)
    att = attention_mask.bool()
    torch.testing.assert_close(step.token_entropy[att], ref_entropy[att], rtol=1e-3, atol=1e-3)
    assert (step.token_entropy[~att] == 0).all()
    torch.testing.assert_close(step.token_loss, ref_token * mask, rtol=1e-4, atol=1e-4)
    assert int(step.correct) == int(ref_correct)
    assert int(step.labelled) == int(mask.sum())


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_inplace_swiglu_matches_pytorch(dtype, shape):
    from solution.mlp import swiglu_inplace

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu_inplace(gate, up)
    out.backward(grad)
    torch.testing.assert_close(out.float(), ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("heads", [12, 2])
def test_rope_matches_pytorch(heads):
    from transformers.models.qwen2.modeling_qwen2 import rotate_half

    from solution.rope import rope_half

    torch.manual_seed(0)
    batch, seq, dim = 3, 50, 128
    # q as Qwen2Attention builds it: a [B, T, H*D] projection viewed as [B, H, T, D]
    q = torch.randn(batch, seq, heads * dim, device="cuda", dtype=torch.float16)
    q = q.view(batch, seq, heads, dim).transpose(1, 2).detach().requires_grad_(True)
    k = torch.randn(batch, heads, seq, dim, device="cuda", dtype=torch.float16, requires_grad=True)
    freqs = torch.rand(1, seq, dim // 2, device="cuda") * 3
    emb = torch.cat([freqs, freqs], -1)
    cos, sin = emb.cos(), emb.sin()
    gq = torch.randn(batch, heads, seq, dim, device="cuda", dtype=torch.float16)
    gk = torch.randn_like(gq)

    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    ref_q = q.float() * c + rotate_half(q.float()) * s
    ref_k = k.float() * c + rotate_half(k.float()) * s
    (ref_q * gq.float()).sum().backward()
    (ref_k * gk.float()).sum().backward()
    ref_dq, ref_dk = q.grad.float(), k.grad.float()
    q.grad = k.grad = None

    with torch.autocast("cuda", dtype=torch.float16):
        out_q, out_k = rope_half(q, k, cos, sin)
    assert out_q.dtype == torch.float16
    (out_q.float() * gq.float()).sum().backward()
    (out_k.float() * gk.float()).sum().backward()
    torch.testing.assert_close(out_q.float(), ref_q, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(out_k.float(), ref_k, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(q.grad.float(), ref_dq, rtol=2e-3, atol=4e-3)
    torch.testing.assert_close(k.grad.float(), ref_dk, rtol=2e-3, atol=4e-3)


def test_lora_add_matches_pytorch():
    from solution.lora import _dropout, _LoraAdd

    torch.manual_seed(0)
    tokens, k_in, n_out, rank, p, scaling, seed = 300, 1536, 256, 16, 0.05, 2.0, 1234
    x = torch.randn(tokens, k_in, device="cuda", dtype=torch.float16, requires_grad=True)
    base = torch.randn(tokens, n_out, device="cuda", dtype=torch.float16, requires_grad=True)
    weight_a = (torch.randn(rank, k_in, device="cuda") * 0.02).requires_grad_(True)
    weight_b = (torch.randn(n_out, rank, device="cuda") * 0.02).requires_grad_(True)
    grad = torch.randn(tokens, n_out, device="cuda", dtype=torch.float16)

    keep = _dropout(torch.ones_like(x), seed, p).float() * (1 - p)
    assert set(keep.unique().round().tolist()) <= {0.0, 1.0}
    assert 0.02 < 1 - keep.mean().item() < 0.08  # about p of the inputs dropped
    ref = base.float() + scaling * ((x.float() * keep / (1 - p)) @ weight_a.t()) @ weight_b.t()
    ref.backward(grad.float())
    refs = [t.grad.float().clone() for t in (x, base, weight_a, weight_b)]
    for t in (x, base, weight_a, weight_b):
        t.grad = None

    out = _LoraAdd.apply(x, base, weight_a, weight_b, p, scaling, seed)
    out.backward(grad)
    torch.testing.assert_close(out.float(), ref, rtol=2e-3, atol=2e-3)
    for got, want in zip((x.grad, base.grad, weight_a.grad, weight_b.grad), refs):
        # fp16 matmuls summing over hundreds of tokens: compare against the tensor's own scale
        scale = want.abs().max().item()
        torch.testing.assert_close(got.float(), want, rtol=1e-2, atol=2e-3 * scale)


@pytest.mark.parametrize("rows", [700, 64])
def test_dequant_once_matches_bitsandbytes(rows):
    import bitsandbytes as bnb

    import solution.dequant as dequant

    torch.manual_seed(0)
    n_out, k_in = 512, 256
    weight = torch.randn(n_out, k_in, device="cuda", dtype=torch.float16)
    packed, state = bnb.functional.quantize_4bit(weight, blocksize=64, compress_statistics=True,
                                                 quant_type="nf4")
    bias = torch.randn(n_out, device="cuda", dtype=torch.float16)
    x = torch.randn(rows, k_in, device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn(rows, n_out, device="cuda", dtype=torch.float16)

    ref = bnb.matmul_4bit(x, packed.t(), quant_state=state, bias=bias)
    ref.backward(grad)
    ref_dx = x.grad.clone()
    x.grad = None

    original = bnb.matmul_4bit
    dequant.patch_bnb_matmul_4bit()
    try:
        out = bnb.matmul_4bit(x, packed.t(), quant_state=state, bias=bias)
        out.backward(grad)
    finally:
        bnb.matmul_4bit = original
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x.grad, ref_dx, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("tokens", [64, 333])
def test_mlp_replay_matches_pytorch(tokens, monkeypatch):
    """The recompute's MLP stand-in (gustash.py): the input gradient of
    down(silu(gate(x)) * up(x)) from the saved gate/up, against plain autograd. Dense fp16 weights
    stand in for the NF4 ones (the dequantization has its own test)."""
    from types import SimpleNamespace

    from solution import gustash

    torch.manual_seed(0)
    hidden, inter = 256, 1000
    w_gate = (torch.randn(inter, hidden, device="cuda") / hidden ** 0.5).half()
    w_up = (torch.randn(inter, hidden, device="cuda") / hidden ** 0.5).half()
    w_down = (torch.randn(hidden, inter, device="cuda") / inter ** 0.5).half()
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn(tokens, hidden, device="cuda", dtype=torch.float16)

    gate, up = x @ w_gate.t(), x @ w_up.t()
    ref = (torch.nn.functional.silu(gate.float()) * up.float()).half() @ w_down.t()
    ref.backward(grad)
    ref_dx = x.grad.float()
    x.grad = None

    monkeypatch.setattr(gustash, "nf4_dequantize", lambda packed, state: state.dense)
    states = [SimpleNamespace(dense=w, shape=tuple(w.shape)) for w in (w_gate, w_up, w_down)]
    packed = [torch.empty(1, device="cuda") for _ in range(3)]
    out = gustash._MlpReplay.apply(x, gate.detach().clone(), up.detach().clone(), *packed, *states)
    assert out.shape == (tokens, hidden) and out.dtype == torch.float16
    out.backward(grad)
    torch.testing.assert_close(x.grad.float(), ref_dx, rtol=2e-2, atol=2e-2)
