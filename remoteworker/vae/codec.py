"""CPU-only, lossless tensor envelopes shared by caller and worker."""
from __future__ import annotations

import io
import torch

MAX_BYTES = 512 * 1024 * 1024


def cpu_value(item):
    if getattr(item, "is_nested", False):
        return {"__remote_vae_nested__": [cpu_value(part) for part in item.unbind()]}
    if torch.is_tensor(item):
        return item.detach().cpu()
    if isinstance(item, dict):
        if "__remote_vae_nested__" in item:
            raise ValueError("Reserved Remote VAE envelope key")
        return {key: cpu_value(part) for key, part in item.items()}
    if isinstance(item, (tuple, list)):
        return type(item)(cpu_value(part) for part in item)
    if item is None or isinstance(item, (str, int, float, bool)):
        return item
    raise ValueError("Unsupported Remote VAE metadata type")

def dumps(value):
    value = cpu_value(value)
    tensors = []

    def visit(item):
        if torch.is_tensor(item):
            if item.is_nested:
                for part in item.unbind():
                    visit(part)
            else:
                tensors.append(item)
        elif isinstance(item, dict):
            for part in item.values():
                visit(part)
        elif isinstance(item, (tuple, list)):
            for part in item:
                visit(part)
    visit(value)
    if sum(t.numel() * t.element_size() for t in tensors) > MAX_BYTES:
        raise ValueError("Remote VAE tensor exceeds the 512 MiB transfer limit")
    stream = io.BytesIO()
    torch.save(value, stream)
    data = stream.getvalue()
    if len(data) > MAX_BYTES:
        raise ValueError("Remote VAE envelope exceeds the 512 MiB transfer limit")
    return data


def restore(item):
    if isinstance(item, dict):
        if set(item) == {"__remote_vae_nested__"}:
            parts = item["__remote_vae_nested__"]
            if not isinstance(parts, list) or not parts or not all(torch.is_tensor(t) for t in parts):
                raise ValueError("Invalid nested latent envelope")
            from comfy.nested_tensor import NestedTensor
            return NestedTensor(parts)
        return {key: restore(part) for key, part in item.items()}
    if isinstance(item, (tuple, list)):
        return type(item)(restore(part) for part in item)
    return item

def loads(data):
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Remote VAE envelope size is invalid")
    return restore(torch.load(io.BytesIO(data), map_location="cpu", weights_only=True))


def validate_latent(value):
    if not isinstance(value, dict) or not (torch.is_tensor(value.get("samples")) or getattr(value.get("samples"), "is_nested", False)):
        raise ValueError("Remote VAE requires a LATENT dictionary with tensor samples")
    return value


def validate_result(value, kind):
    if kind == "image":
        valid = torch.is_tensor(value) and value.ndim == 4 and value.shape[-1] in (3, 4)
    else:
        valid = (isinstance(value, dict) and torch.is_tensor(value.get("waveform"))
                 and value["waveform"].ndim == 3 and isinstance(value.get("sample_rate"), int)
                 and value["sample_rate"] > 0)
    if not valid:
        raise ValueError(f"Remote VAE returned invalid {kind} data")
    return value
