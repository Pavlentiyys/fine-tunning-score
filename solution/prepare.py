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
NO_ALLOCATOR_WARMUP = True
VOCAB, HIDDEN = 151936, 1536  # Qwen2.5-1.5B's tied embedding: the one tensor of this shape

# what happened to the embedding while loading; printed with the first [mem] line of training
LOAD_EVENTS = {"materialize calls": 0, "embedding materialized on cpu": 0,
               "dispatch kept it on cpu": 0, "dispatch saw it on cuda": 0}


def load_summary():
    return ", ".join(f"{name} {count}" for name, count in LOAD_EVENTS.items())


def patch_no_allocator_warmup():
    """Skip transformers' `caching_allocator_warmup`.

    Before loading the weights, transformers allocates (and frees) one tensor as large as all the
    parameters bound for the GPU, so that the caching allocator reserves that memory in one cudaMalloc
    instead of many. Here the 4-bit weights leave the GPU as soon as they are quantized (stream.py)
    and the embedding is loaded to the CPU, so that reservation (~1.07 GB: the model's full size) was
    the peak of the whole process, held by the allocator although nothing ever used it at once. Without
    it, loading allocates per tensor, which on a 1.5B 4-bit model costs a fraction of a second."""
    if not NO_ALLOCATOR_WARMUP:
        return
    from transformers import modeling_utils

    def caching_allocator_warmup(*args, **kwargs):
        print("[load] transformers' caching_allocator_warmup skipped", flush=True)

    modeling_utils.caching_allocator_warmup = caching_allocator_warmup


def patch_embedding_load_to_cpu():
    if not EMBEDDING_ON_CPU:
        return
    from transformers import core_model_loading
    from transformers.modeling_utils import PreTrainedModel

    original = core_model_loading._materialize_copy

    def materialize(tensor, device=None, dtype=None):
        # `tensor` may be a lazy safetensors slice without `.shape`: read it first (on the CPU,
        # which is where the slice is read anyway), then decide where it goes
        LOAD_EVENTS["materialize calls"] += 1
        tensor = tensor[...]
        if (device is not None and torch.device(device).type == "cuda"
                and tuple(tensor.shape) == (VOCAB, HIDDEN)):
            LOAD_EVENTS["embedding materialized on cpu"] += 1
            print(f"[load] tied embedding kept in CPU memory (was bound for {device})", flush=True)
            return tensor.to(dtype=dtype or torch.float16)
        return original(tensor, device, dtype)

    core_model_loading._materialize_copy = materialize

    # After loading, transformers calls accelerate's `dispatch_model`, which for a single-GPU device
    # map runs `model.to(device)` and would carry the CPU embedding to the GPU after all. The
    # embedding's data is swapped for an empty CPU tensor during that call and put back after.
    from transformers import modeling_utils
    from transformers.integrations import accelerate as hf_accelerate

    dispatch = hf_accelerate.accelerate_dispatch

    def accelerate_dispatch(model, *args, **kwargs):
        weight = getattr(model.get_input_embeddings(), "weight", None)
        if weight is None or weight.is_cuda:
            if weight is not None:
                LOAD_EVENTS["dispatch saw it on cuda"] += 1
            return dispatch(model, *args, **kwargs)
        LOAD_EVENTS["dispatch kept it on cpu"] += 1
        kept = weight.data
        weight.data = torch.empty(0, dtype=kept.dtype)
        try:
            return dispatch(model, *args, **kwargs)
        finally:
            weight.data = kept
            print(f"[load] embedding kept on {weight.device} through dispatch_model", flush=True)

    hf_accelerate.accelerate_dispatch = accelerate_dispatch
    modeling_utils.accelerate_dispatch = accelerate_dispatch

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
