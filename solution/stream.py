"""Layer-wise weight streaming: the 4-bit weights of the decoder layers live in pinned CPU memory
and are copied to the GPU one layer at a time, with the next layer's copy running on a side stream
while the current layer computes.

Why: the 28 layers' NF4 weights are ~655 MB of the ~1.2 GB that are left on the GPU once the
activations are recomputed. A layer's packed weights are ~23 MB; the peak is the working set of
two layers (the one computing and the one being fetched) instead of all of them.

Loading is intercepted where bitsandbytes quantizes a weight (`Params4bit._quantize`, called while
transformers builds the model): right after a weight is quantized on the GPU its packed data and
quantization statistics are copied to pinned CPU memory and the GPU keeps empty placeholders, so
the whole model is never on the GPU at once and the process peak stays low from the first second.

A decoder layer then runs inside `StreamedLayer`, a custom autograd function: forward fetches the
layer, runs it and drops it, keeping only the layer input; backward fetches it again, recomputes
the layer with gradients enabled and backpropagates through it (also the LoRA weights' gradients),
then drops it. That is the same recomputation as before, with the weight copy around it.

Not done here: Soup's `gradient_checkpointing` switch (stays off), a second GPU, anything outside
this process. If a weight cannot be mapped to its layer, everything is copied back to the GPU and
streaming is switched off, so a surprise costs memory, not correctness.
"""

import torch

ENABLED = True
OFFLOAD_INPUTS = True  # the layer inputs kept for the backward also go to pinned CPU memory
PREFETCH_DEPTH = 2
ALIGN = 256  # byte alignment of each tensor inside a layer's packed buffer
# the packed weights and the per-block scales; `offset` (one scalar) and `code` (256 values) stay on
# the GPU: bitsandbytes reads them with .item() when it serializes the quantization state
_SLOT_ATTRS = ("absmax",)


class _Slot:
    """One tensor attribute (`owner.attr`) whose data is kept on the CPU. Until the first forward
    the copy is in ordinary (pageable) memory: pinning ~600 small tensors one by one while the model
    loads was slow. The first forward packs each layer's slots into one pinned arena."""

    __slots__ = ("owner", "attr", "cpu", "device", "placeholder", "dtype", "shape", "nbytes",
                 "offset")

    def __init__(self, owner, attr, tensor):
        self.owner, self.attr, self.device = owner, attr, tensor.device
        self.dtype, self.shape = tensor.dtype, tuple(tensor.shape)
        self.nbytes, self.offset = tensor.numel() * tensor.element_size(), None
        self.cpu = tensor.detach().to("cpu")
        self.placeholder = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
        self.drop()

    def drop(self):
        if self.attr == "data":
            self.owner.data = self.placeholder
        else:
            setattr(self.owner, self.attr, self.placeholder)

    def set(self, tensor):
        if self.attr == "data":
            self.owner.data = tensor
        else:
            setattr(self.owner, self.attr, tensor)

    def view_in(self, buffer):
        """This slot's tensor as a view into a layer's packed uint8 buffer."""
        raw = buffer[self.offset:self.offset + self.nbytes]
        return raw.view(self.dtype).view(self.shape)


_REGISTRY = {}  # id(Params4bit) -> (param, [slots])
_STATE = {"manager": None, "disabled": False}


def offload_param(param):
    """Move a freshly quantized Params4bit's tensors to CPU memory."""
    if id(param) in _REGISTRY or param.quant_state is None or not param.data.is_cuda:
        return
    slots = [_Slot(param, "data", param.data)]
    state = param.quant_state
    for owner in (state, getattr(state, "state2", None)):
        if owner is None:
            continue
        for attr in _SLOT_ATTRS:
            value = getattr(owner, attr, None)
            if torch.is_tensor(value) and value.is_cuda:
                slots.append(_Slot(owner, attr, value))
    _REGISTRY[id(param)] = (param, slots)


def restore_all():
    """Copy every offloaded tensor back to the GPU and stop streaming (the safe way out)."""
    _STATE["disabled"] = True
    manager = _STATE["manager"]
    if manager is not None:  # packed: each layer's slots are views into its span of the arena
        for slots, (start, stop) in zip(manager.groups, manager.spans):
            span = manager.arena[start:stop].to("cuda")
            for slot in slots:
                slot.set(slot.view_in(span).clone())
    else:
        for _param, slots in _REGISTRY.values():
            for slot in slots:
                slot.set(slot.cpu.to(slot.device))
    _REGISTRY.clear()
    _STATE["manager"] = None


class LayerManager:
    """Fetches and drops the offloaded weights of one decoder layer at a time: one pinned arena
    for all layers, one host-to-device copy per layer, views into it for each tensor."""

    def __init__(self, layers):
        self.groups, self.spans = [], []
        total = 0
        for layer in layers:
            slots = []
            for param in layer.parameters():
                if param.__class__.__name__ == "Params4bit":
                    entry = _REGISTRY.get(id(param))
                    if entry is None:
                        raise LookupError("a 4-bit weight was not offloaded")
                    slots += entry[1]
            start = total
            for slot in slots:
                slot.offset = total - start
                total += -(-slot.nbytes // ALIGN) * ALIGN
            self.groups.append(slots)
            self.spans.append((start, total))
        self.arena = torch.empty(total, dtype=torch.uint8, pin_memory=True)
        for slots, (start, _stop) in zip(self.groups, self.spans):
            for slot in slots:
                self.arena[start + slot.offset:start + slot.offset + slot.nbytes].copy_(
                    slot.cpu.reshape(-1).view(torch.uint8))
                slot.cpu = None
        self.side = torch.cuda.Stream()
        self.staged = {}

    def prefetch(self, index):
        if not 0 <= index < len(self.groups) or index in self.staged or not self.groups[index]:
            return
        start, stop = self.spans[index]
        with torch.cuda.stream(self.side):
            buffer = torch.empty(stop - start, dtype=torch.uint8, device="cuda")
            buffer.copy_(self.arena[start:stop], non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.side)
        self.staged[index] = (buffer, event)

    def fetch(self, index, step):
        if index not in self.staged:
            self.prefetch(index)
        buffer, event = self.staged.pop(index)
        stream = torch.cuda.current_stream()
        stream.wait_event(event)
        buffer.record_stream(stream)
        for slot in self.groups[index]:
            slot.set(slot.view_in(buffer))
        for ahead in range(1, PREFETCH_DEPTH + 1):
            self.prefetch(index + step * ahead)

    def release(self, index):
        for slot in self.groups[index]:
            slot.drop()

    def to_host(self, tensor):
        """Async copy of a layer input to pinned memory on the side stream."""
        host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
        done = torch.cuda.Event()
        self.side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.side):
            host.copy_(tensor, non_blocking=True)
            done.record(self.side)
        tensor.record_stream(self.side)
        return host, done

    def from_host(self, host, done):
        stream = torch.cuda.current_stream()
        stream.wait_event(done)
        return host.to("cuda", non_blocking=True)


def get_manager(layers):
    """The manager, built once on the first forward; None when streaming is off."""
    if _STATE["disabled"] or not ENABLED or not _REGISTRY:
        return None
    if _STATE["manager"] is None:
        try:
            _STATE["manager"] = LayerManager(layers)
        except LookupError:
            restore_all()
            return None
    return _STATE["manager"]


def streaming():
    return _STATE["manager"] is not None


class _Streamed(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, hidden, holder, *params):
        manager, index = holder.manager, holder.index
        manager.fetch(index, +1)
        if index + 1 == len(manager.groups):
            manager.prefetch(index)  # the backward starts with this layer
        # grad mode on, as in a recomputed layer: the patched LoRA / RoPE / norm paths are the
        # ones that need it; the graph this builds is dropped with `out.detach()`
        with torch.enable_grad():
            out = holder.run(hidden)
        manager.release(index)
        ctx.holder, ctx.needs_grad = holder, hidden.requires_grad
        if OFFLOAD_INPUTS:
            ctx.host = manager.to_host(hidden.detach())
        else:
            ctx.host = None
            ctx.save_for_backward(hidden)
        return out.detach()

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        holder = ctx.holder
        manager, index = holder.manager, holder.index
        if ctx.host is not None:
            hidden = manager.from_host(*ctx.host)
            ctx.host = None
        else:
            (hidden,) = ctx.saved_tensors
        manager.fetch(index, -1)
        with torch.enable_grad():
            states = hidden.detach().requires_grad_(ctx.needs_grad)
            out = holder.run(states)
            params = holder.params
            inputs = ([states] if states.requires_grad else []) + params
            grads = torch.autograd.grad(out, inputs, grad_out, allow_unused=True)
        manager.release(index)
        del out
        grad_hidden = grads[0] if states.requires_grad else None
        param_grads = grads[1:] if states.requires_grad else grads
        return (grad_hidden, None, *param_grads)


class LayerCall:
    """What one streamed layer call needs: how to run the layer and which weights get gradients."""

    def __init__(self, manager, index, run, params):
        self.manager, self.index, self.run, self.params = manager, index, run, params


def streamed_layer(manager, index, run, layer, hidden):
    params = [p for p in layer.parameters() if p.requires_grad]
    holder = LayerCall(manager, index, run, params)
    return _Streamed.apply(hidden, holder, *params)


def patch_bnb_offload():
    """Offload every weight right after bitsandbytes quantizes it."""
    if not ENABLED or not torch.cuda.is_available():
        return
    import bitsandbytes as bnb

    original = bnb.nn.Params4bit._quantize

    def _quantize(self, device):
        result = original(self, device)
        if not _STATE["disabled"] and torch.device(device).type == "cuda":
            offload_param(result)
        return result

    bnb.nn.Params4bit._quantize = _quantize
