from __future__ import annotations

import torch


def discover_device_options() -> list[tuple[str, str]]:
    """
    Returns a list of (value, label), e.g.
    ("cpu", "CPU")
    ("cuda:0", "CUDA 0 — NVIDIA RTX 4090")
    """
    options: list[tuple[str, str]] = [("cpu", "CPU")]

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            options.append((f"cuda:{i}", f"CUDA {i} — {name}"))

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        options.append(("mps", "Apple Metal (MPS)"))

    return options


def default_device_value(options: list[tuple[str, str]]) -> str:
    values = [value for value, _ in options]
    if "cuda:0" in values:
        return "cuda:0"
    if "mps" in values:
        return "mps"
    return "cpu"