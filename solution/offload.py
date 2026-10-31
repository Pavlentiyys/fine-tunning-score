"""Streamed layers without recomputation: the activations a layer saves for its backward go to
pinned CPU memory and come back, one layer ahead, during the backward.

With weights streamed (stream.py) every layer used to run forward twice: once without a graph,
and again in the backward to rebuild the graph it backpropagates through. Here the forward of a
streamed layer builds its autograd graph normally, under `saved_tensors_hooks`:

- every saved tensor of 1 MiB or more (layer inputs to the norms and LoRA, gate/up, q/k/v/out of
  the attention) is copied to pinned host memory on the stream manager's side stream right after
  the layer's forward, and its GPU memory is released once the copy is done;
- the 4-bit weights the matmuls keep (views into the manager's ring buffers) are kept as they are:
  the backward of the layer fetches the layer's weights into the same buffer first, so the views
  hold the right data again; small tensors (LoRA factors, scales, row statistics) stay on the GPU;
- the last layer's activations are not moved at all: its backward comes right after the head.

The backward of layer i starts when the gradient of its output is ready (a tensor hook): the hook
releases layer i+1's weight buffer (its backward is done), fetches layer i's weights, and starts
copying layer i-1's activations back on the side stream, so they are on the GPU before autograd
asks for them. Same math as the recomputation (same kernels, same dropout seeds), one forward pass
less per step. Layer 0's buffer is released at the end of the training step.
"""

import torch

ENABLED = True  # False: streamed layers recompute in the backward (stream._Streamed) as before
MIN_BYTES = 1 << 20  # saved tensors from this size go to the host; smaller ones stay on the GPU
MAX_PENDING = 1  # layers whose activations may still be on the GPU waiting for their host copy


def _dense(t):
    """Non-overlapping and dense, so empty_strided can hold exactly the same layout."""
    if t.numel() == 0:
        return False
    if any(st < 0 for st in t.stride()):
        return False
    span = 1 + sum((size - 1) * st for size, st in zip(t.shape, t.stride()) if size > 1)
    return span == t.numel()


class _Saved:
    __slots__ = ("gpu", "host", "done", "back", "ready", "uses")

    def __init__(self, gpu):
        self.gpu, self.host, self.done, self.back, self.ready = gpu, None, None, None, None
        self.uses = 1


class Offloader:
    def __init__(self, manager):
        self.manager = manager
        self.side = manager.side  # host -> device: weights and the activations coming back
        self.d2h = torch.cuda.Stream()  # device -> host: its own copy engine, next to the weights
        self.layers = {}  # layer -> [_Saved] of its forward
        self.pending = []  # (gpu tensors, event): kept alive until their copy to the host is done
        self.held = set()  # layers whose weight buffer is still held after their forward
        self.last = len(manager.groups) - 1

    def _ring_ptrs(self):
        return {buffer.untyped_storage().data_ptr() for buffer in self.manager.ring}

    # ------------------------------------------------------------------ forward side
    def pack_for(self, index):
        saved = self.layers.setdefault(index, [])
        ring = self._ring_ptrs()
        keep_all = index == self.last
        seen = {}  # the same tensor saved by two nodes is copied once

        def pack(t):
            if (keep_all or not t.is_cuda or t.untyped_storage().nbytes() < MIN_BYTES
                    or t.untyped_storage().data_ptr() in ring):
                return t
            key = (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)
            entry = seen.get(key)
            if entry is not None and entry.gpu is not None:
                entry.uses += 1
                return entry
            entry = seen[key] = _Saved(t)
            saved.append(entry)
            return entry

        return pack

    def flush(self, index):
        """Copy the layer's large saved tensors to pinned host memory (side stream)."""
        saved = self.layers.get(index)
        if not saved:
            return
        stream = torch.cuda.current_stream()
        self.d2h.wait_stream(stream)
        tensors = []
        with torch.cuda.stream(self.d2h), torch.no_grad():
            for entry in saved:
                gpu = entry.gpu
                if _dense(gpu):
                    host = torch.empty_strided(gpu.shape, gpu.stride(), dtype=gpu.dtype,
                                               pin_memory=True)
                else:
                    host = torch.empty(gpu.shape, dtype=gpu.dtype, pin_memory=True)
                host.copy_(gpu, non_blocking=True)
                entry.host = host
                tensors.append(gpu)
            done = torch.cuda.Event()
            done.record(self.d2h)
        for entry in saved:
            entry.done, entry.gpu = done, None
        self.pending.append((tensors, done))
        while len(self.pending) > MAX_PENDING:
            # the copies are behind the compute: let the compute wait for the oldest layer's
            # copy, so its GPU memory is reused only after the copy has read it
            old_tensors, old_done = self.pending.pop(0)
            stream.wait_event(old_done)
            del old_tensors

    def drain(self):
        stream = torch.cuda.current_stream()
        for tensors, done in self.pending:
            stream.wait_event(done)
        self.pending = []

    def prune(self):
        self.pending = [(tensors, done) for tensors, done in self.pending if not done.query()]

    # ------------------------------------------------------------------ backward side
    def prefetch(self, index):
        """Start copying layer `index`'s saved tensors back to the GPU on the side stream."""
        saved = self.layers.get(index)
        if not saved:
            return
        todo = [entry for entry in saved if entry.back is None and entry.host is not None]
        if not todo:
            return
        stream = torch.cuda.current_stream()
        backs = []
        for entry in todo:  # allocated on the main stream, filled on the side stream
            host = entry.host
            backs.append(torch.empty_strided(host.shape, host.stride(), dtype=host.dtype,
                                             device="cuda"))
        self.side.wait_stream(stream)
        with torch.cuda.stream(self.side):
            for entry, back in zip(todo, backs):
                self.side.wait_event(entry.done)
                back.copy_(entry.host, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self.side)
        for entry, back in zip(todo, backs):
            entry.back, entry.ready = back, ready

    def unpack(self, entry):
        if not isinstance(entry, _Saved):
            return entry
        stream = torch.cuda.current_stream()
        if entry.back is None:  # not prefetched: copy now on the main stream
            stream.wait_event(entry.done)
            back = torch.empty_strided(entry.host.shape, entry.host.stride(),
                                       dtype=entry.host.dtype, device="cuda")
            back.copy_(entry.host, non_blocking=True)
            entry.uses -= 1
            if entry.uses > 0:
                entry.back, entry.ready = back, None
            else:
                entry.host = None
            return back
        if entry.ready is not None:
            stream.wait_event(entry.ready)
        back = entry.back
        entry.uses -= 1
        if entry.uses <= 0:
            entry.back, entry.host = None, None
        return back

    def backward_start(self, index):
        """Gradient of layer `index`'s output is ready: its backward is about to run."""
        manager = self.manager
        self.prune()
        if index + 1 in self.held:  # layer index+1 has finished its backward
            self.held.discard(index + 1)
            manager.release(index + 1)
        manager.fetch(index, -1)
        self.held.add(index)
        self.prefetch(index)  # normally already started one layer earlier
        self.prefetch(index - 1)
        # layer index+1's backward is over: whatever of it is still on the GPU (a saved tensor
        # whose node did not run, or did not ask for it) is dropped now, not at the end of the step
        for entry in self.layers.pop(index + 1, ()):
            entry.back, entry.host, entry.gpu = None, None, None

    def step_end(self):
        """After the step's backward: release what is still held, clear the bookkeeping."""
        for index in sorted(self.held):
            self.manager.release(index)
        self.held.clear()
        for saved in self.layers.values():
            for entry in saved:
                entry.back, entry.host, entry.gpu = None, None, None
        self.layers.clear()
        self.prune()


_STATE = {"offloader": None}


def get_offloader(manager):
    off = _STATE["offloader"]
    if off is None or off.manager is not manager:
        off = _STATE["offloader"] = Offloader(manager)
    return off


def offloaded_layer(manager, index, run, hidden):
    """Run one streamed decoder layer with its activations offloaded instead of recomputed."""
    off = get_offloader(manager)
    off.prune()
    manager.fetch(index, +1)
    with torch.autograd.graph.saved_tensors_hooks(off.pack_for(index), off.unpack):
        out = run(hidden)
    off.flush(index)
    last = index == off.last
    if last:  # nothing more comes before the backward: let the copies finish, free their GPU memory
        off.drain()
    manager.release(index, keep=last)
    if last:
        off.held.add(index)
    if out.requires_grad:
        out.register_hook(lambda grad, index=index: off.backward_start(index))
    return out


def patch_step_end():
    """Release the last held weight buffer after each training step's backward."""
    try:
        import transformers
    except Exception:  # noqa: BLE001 - no transformers (fake modules in tests)
        return
    original = transformers.Trainer.training_step

    def training_step(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        finally:
            off = _STATE["offloader"]
            if off is not None:
                off.step_end()

    transformers.Trainer.training_step = training_step
