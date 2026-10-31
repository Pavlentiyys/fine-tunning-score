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


@pytest.mark.parametrize("vocab", [151936, 5003])
def test_fused_head_loss_matches_pytorch(vocab):
    from solution.fused_ce import fused_head_loss

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
