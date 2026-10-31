"""Start-up time outside the training loop. Nothing about training, data or outputs is touched; if
a patch cannot be installed, the original function stays in place.

`packages_distributions()` walks every installed distribution's metadata (a few thousand files in
Kaggle's image) and several libraries that `soup train` imports (transformers, peft, accelerate,
Soup itself) each ask for it. The answer cannot change inside one process, so the first call is
kept and later calls get a copy of it.

`Dataset.map` fingerprints the mapped function by pickling it with dill. Soup tokenizes the rows
with a closure that holds the whole tokenizer (`_map_text_sft_rows`), which makes that hash cost
seconds. The dataset is in memory (`Dataset.from_list`), so no cache file is looked up by the
fingerprint; a fresh random one gives the same rows. Soup's own rewind fingerprint is a sha256 of
the raw rows and does not read it.
"""

import functools
import importlib.metadata as metadata
import inspect
import uuid

_CACHE = []


def patch_packages_distributions():
    original = getattr(metadata, "packages_distributions", None)
    if original is None or getattr(original, "_soup_cached", False):
        return

    def packages_distributions():
        if not _CACHE:
            _CACHE.append({name: list(dists) for name, dists in original().items()})
        return {name: list(dists) for name, dists in _CACHE[0].items()}

    packages_distributions._soup_cached = True
    metadata.packages_distributions = packages_distributions


def patch_dataset_map_fingerprint():
    try:
        from datasets import Dataset
    except Exception:  # noqa: BLE001 - no datasets (tests with fake modules): nothing to patch
        return
    original = Dataset.map
    if getattr(original, "_soup_fingerprint", False):
        return
    try:
        if "new_fingerprint" not in inspect.signature(original).parameters:
            return
    except (TypeError, ValueError):
        return

    @functools.wraps(original)
    def map(self, *args, **kwargs):
        if kwargs.get("new_fingerprint") is None and not getattr(self, "cache_files", None):
            kwargs["new_fingerprint"] = uuid.uuid4().hex
        return original(self, *args, **kwargs)

    map._soup_fingerprint = True
    Dataset.map = map


# GPU memory outside PyTorch's allocator (nvidia-smi counts it, torch does not):
# - the CUDA driver reserves the per-thread stack for every resident GPU thread, 1 KiB each by
#   default (~40 MiB on a T4); it grows the stack by itself for a kernel that needs more;
# - every thread that calls cuBLAS gets its own workspace (~8 MiB each by default). `:16:8`
#   (eight 16 KiB chunks) is one of the two sizes cuBLAS documents for reproducible results.
CUDA_STACK_BYTES = 16
CUBLAS_WORKSPACE = ":16:8"


def shrink_cuda_reserves():
    import os

    if CUBLAS_WORKSPACE and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE  # read at the first cuBLAS call
    if not CUDA_STACK_BYTES:
        return
    try:
        import ctypes

        import torch

        if not torch.cuda.is_available():
            return
        torch.cuda.init()  # the harness made the context current already; this is a no-op then
        driver = ctypes.CDLL("libcuda.so.1")
        before = ctypes.c_size_t(0)
        driver.cuCtxGetLimit(ctypes.byref(before), 0)  # CU_LIMIT_STACK_SIZE
        status = driver.cuCtxSetLimit(0, ctypes.c_size_t(CUDA_STACK_BYTES))
        after = ctypes.c_size_t(0)
        driver.cuCtxGetLimit(ctypes.byref(after), 0)
        print(f"[load] CUDA stack per thread {before.value} -> {after.value} bytes "
              f"(status {status}); cuBLAS workspace {os.environ.get('CUBLAS_WORKSPACE_CONFIG')}",
              flush=True)
    except Exception as exc:  # noqa: BLE001 - the defaults stay
        print(f"[load] CUDA stack limit unchanged: {exc!r}", flush=True)
