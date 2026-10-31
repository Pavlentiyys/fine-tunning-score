"""Keep the frozen tied embedding in fp16 from the start, instead of fp32.

`peft.prepare_model_for_kbit_training` (called by Soup) casts every fp16 parameter that is not
a 4-bit weight to fp32. For the tied embedding / lm_head (151936 x 1536) that means a 934 MB fp32
copy next to the 467 MB fp16 one for a few milliseconds, which was the allocator's peak but shorter
than nvidia-smi's 100 ms sampling (check 9 compares the two), and then 934 MB that stay on the GPU.
The head runs in fp16 under autocast anyway and the residual stream is made fp32 after the lookup
(see unpad.py), so nothing needs the fp32 copy.

The call is wrapped: the embedding's data is swapped for an empty tensor while peft casts, and put
back (still fp16) afterwards. The weights, the trainable set and everything else are untouched.
"""

import torch


def patch_kbit_preparation():
    import peft
    from peft.utils import other

    original = other.prepare_model_for_kbit_training

    def prepare(model, *args, **kwargs):
        embedding = None
        try:
            embedding = model.get_input_embeddings()
        except Exception:  # noqa: BLE001 - a model without one is simply not touched
            pass
        weight = getattr(embedding, "weight", None)
        if weight is None or weight.dtype != torch.float16 or not weight.is_cuda:
            return original(model, *args, **kwargs)
        kept = weight.data
        weight.data = torch.empty(0, dtype=torch.float16, device=kept.device)
        try:
            model = original(model, *args, **kwargs)
        finally:
            weight.data = kept
        return model

    other.prepare_model_for_kbit_training = prepare
    peft.prepare_model_for_kbit_training = prepare
