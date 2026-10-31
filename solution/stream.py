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
# the packed weights and the per-block scales; `offset` (one scalar) and `code` (256 values) stay on
# the GPU: bitsandbytes reads them with .item() when it serializes the quantization state
_SLOT_ATTRS = ("absmax",)


class _Slot:
    """One tensor attribute (`owner.attr`) whose data is kept on the CPU."""

    __slots__ = ("owner", "attr", "cpu", "device", "placeholder")

    def __init__(self, owner, attr, tensor):
        self.owner, self.attr, self.device = owner, attr, tensor.device
        self.cpu = tensor.detach().to("cpu").pin_memory()
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


_REGISTRY = {}  # id(Params4bit) -> (param, [slots])
_STATE = {"manager": None, "disabled": False}


def offload_param(param):
    """Move a freshly quantized Params4bit's tensors to pinned CPU memory."""
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
    for _param, slots in _REGISTRY.values():
        for slot in slots:
            slot.set(slot.cpu.to(slot.device))
    _REGISTRY.clear()


class LayerManager:
    """Fetches and drops the offloaded weights of one decoder layer at a time."""

    def __init__(self, layers):
        self.groups = []
        for layer in layers:
            slots = []
            for param in layer.parameters():
                if param.__class__.__name__ == "Params4bit":
                    entry = _REGISTRY.get(id(param))
                    if entry is None:
                        raise LookupError("a 4-bit weight was not offloaded")
                    slots += entry[1]
            self.groups.append(slots)
        self.side = torch.cuda.Stream()
        self.staged = {}

    def prefetch(self, index):
        if not 0 <= index < len(self.groups) or index in self.staged or not self.groups[index]:
            return
        with torch.cuda.stream(self.side):
            tensors = [slot.cpu.to(slot.device, non_blocking=True) for slot in self.groups[index]]
            event = torch.cuda.Event()
            event.record(self.side)
        self.staged[index] = (tensors, event)

    def fetch(self, index):
        if index not in self.staged:
            self.prefetch(index)
        tensors, event = self.staged.pop(index)
        stream = torch.cuda.current_stream()
        stream.wait_event(event)
        for slot, tensor in zip(self.groups[index], tensors):
            tensor.record_stream(stream)
            slot.set(tensor)

    def release(self, index):
        for slot in self.groups[index]:
            slot.drop()


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


class _Streamed(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, hidden, holder, *params):
        manager, index = holder.manager, holder.index
        manager.fetch(index)
        manager.prefetch(index + 1 if index + 1 < len(manager.groups) else index)
        # grad mode on, as in a recomputed layer: the patched LoRA / RoPE / norm paths are the
        # ones that need it; the graph this builds is dropped with `out.detach()`
        with torch.enable_grad():
            out = holder.run(hidden)
        manager.release(index)
        ctx.holder = holder
        ctx.save_for_backward(hidden)
        return out.detach()

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        (hidden,) = ctx.saved_tensors
        holder = ctx.holder
        manager, index = holder.manager, holder.index
        manager.fetch(index)
        manager.prefetch(index - 1)
        with torch.enable_grad():
            states = hidden.detach().requires_grad_(hidden.requires_grad)
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
