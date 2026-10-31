"""Keep the frozen tied embedding in fp16 from the start, instead of fp32.

`peft.prepare_model_for_kbit_training` (called by Soup) casts every fp16 parameter that is not
a 4-bit weight to fp32. For the tied embedding / lm_head (151936 x 1536) that means a 934 MB fp32
copy next to the 467 MB fp16 one for a few milliseconds, which was the allocator's peak but shorter
than nvidia-smi's 100 ms sampling (check 9 compares the two), and then 934 MB that stay on the GPU.
The head runs in fp16 under autocast anyway and the residual stream is made fp32 after the lookup
(see unpad.py), so nothing needs the fp32 copy.

The call is wrapped: the embedding's data is swapped for an empty tensor while peft casts, and put
back (still fp16) afterwards. The weights, the trainable set and everything else are untouched.

Further, with EMBEDDING_ON_CPU the tied embedding / lm_head never reaches the GPU: transformers
materializes each checkpoint tensor with `tensor.to(device, dtype)` (for the bf16 -> fp16 embedding
that briefly held both copies on the GPU, the process peak during loading), and this one tensor is
materialized on the CPU instead. The training forward then looks up the real tokens' rows on the
CPU (unpad.py) and streams the head in vocabulary chunks (fused_ce.py).
"""

import torch

EMBEDDING_ON_CPU = True
VOCAB, HIDDEN = 151936, 1536  # Qwen2.5-1.5B's tied embedding: the one tensor of this shape


def patch_embedding_load_to_cpu():
    if not EMBEDDING_ON_CPU:
        return
    from transformers import core_model_loading
    from transformers.modeling_utils import PreTrainedModel

    original = core_model_loading._materialize_copy

    def materialize(tensor, device=None, dtype=None):
        # `tensor` may be a lazy safetensors slice without `.shape`: read it first (on the CPU,
        # which is where the slice is read anyway), then decide where it goes
        tensor = tensor[...]
        if (device is not None and torch.device(device).type == "cuda"
                and tuple(tensor.shape) == (VOCAB, HIDDEN)):
            print("[load] tied embedding kept in CPU memory", flush=True)
            return tensor.to(dtype=dtype or torch.float16)
        return original(tensor, device, dtype)

    core_model_loading._materialize_copy = materialize

    # `model.device` is the first parameter's device, which is now the CPU embedding; anything
    # that places inputs by it must still see the GPU the rest of the model is on
    device_property = PreTrainedModel.device

    def device(self):
        found = device_property.fget(self)
        if found.type == "cpu":
            for param in self.parameters():
                if param.is_cuda:
                    return param.device
        return found

    PreTrainedModel.device = property(device)


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
        if weight is None or weight.dtype != torch.float16:
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
