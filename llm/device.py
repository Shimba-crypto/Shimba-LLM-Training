"""
device.py — Device resolution for Shimba LLM.

Runs on whatever PyTorch gives us: CUDA on Colab / Kaggle / a local GPU box,
MPS on Apple Silicon, CPU everywhere else. Nothing here touches the global
default device, so importing this package has no side effects.
"""

from __future__ import annotations

import torch

AUTO = "auto"


def cuda_available() -> bool:
    return torch.cuda.is_available()


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend and backend.is_available())


def resolve_device(spec: str | None = AUTO) -> torch.device:
    """
    Turn a device string into a concrete torch.device.

    Accepts: "auto" | "cpu" | "cuda" | "cuda:1" | "mps".
    Falls back to CPU when the requested accelerator is unavailable, so a
    missing GPU degrades performance instead of raising.
    """
    if spec is None or spec == AUTO:
        if cuda_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    dev = torch.device(spec)

    if dev.type == "cuda" and not cuda_available():
        print("[device] CUDA requested but unavailable — falling back to CPU")
        return torch.device("cpu")

    if dev.type == "mps" and not mps_available():
        print("[device] MPS requested but unavailable — falling back to CPU")
        return torch.device("cpu")

    return dev


def describe_device(device: torch.device) -> str:
    """Human-readable one-liner for the training banner."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    if device.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"


def dtype_for(device: torch.device, prefer_bf16: bool = True) -> torch.dtype:
    """
    Training dtype for a device.

    float32 on CPU: fp16/bf16 matmuls on CPU are slower or unsupported.
    bfloat16 on CUDA when the GPU supports it, otherwise float16.
    """
    if device.type != "cuda":
        return torch.float32
    if prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_enabled(device: torch.device, prefer_bf16: bool = True) -> bool:
    """Whether to wrap forward passes in torch.autocast on this device."""
    if device.type == "cuda":
        return True
    if device.type == "mps" and prefer_bf16:
        return True
    return False