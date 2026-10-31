"""Start-up time outside the training loop: `importlib.metadata.packages_distributions()` once.

`packages_distributions()` walks every installed distribution's metadata (a few thousand files in
Kaggle's image) and several libraries that `soup train` imports (transformers, peft, accelerate,
Soup itself) each ask for it. The answer cannot change inside one process, so the first call is
kept and later calls get a copy of it. Nothing about training, data or outputs is touched; if the
patch cannot be installed, the original function stays in place.
"""

import importlib.metadata as metadata

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
