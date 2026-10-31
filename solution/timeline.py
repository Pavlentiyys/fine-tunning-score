"""Timestamps in train.log for the parts of `soup train` outside the training loop (read only).

The harness times the whole command and the loop; this splits the rest: model loading, the
trainer's setup, the loop, and saving at the end. It only prints, it changes nothing.
"""

import atexit
import time

_T0 = time.perf_counter()


def mark(label):
    print(f"[time] {label}: {time.perf_counter() - _T0:.2f} s after apply()", flush=True)


def patch_timeline():
    import transformers

    auto = transformers.AutoModelForCausalLM
    load = auto.from_pretrained.__func__

    def from_pretrained(cls, *args, **kwargs):
        mark("model load start")
        model = load(cls, *args, **kwargs)
        mark("model load end")
        return model

    auto.from_pretrained = classmethod(from_pretrained)

    train = transformers.Trainer.train

    def timed_train(self, *args, **kwargs):
        mark("trainer.train start")
        try:
            return train(self, *args, **kwargs)
        finally:
            mark("trainer.train end")

    transformers.Trainer.train = timed_train
    atexit.register(mark, "process exit")
