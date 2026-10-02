"""
checkpoint.py — Canonical, backwards-compatible checkpoint I/O.

Older versions of this project wrote weights under three different keys
("state_dict", "model", "weights") and could emit `_orig_mod.`-prefixed keys
whenever `torch.compile` was active. Both problems made `load_state_dict`
fail. Everything funnels through here now.

Pure helpers, no torch.compile awareness leaks elsewhere. `GPTConfig` is
imported lazily inside the load path to avoid a circular import with model.py.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

# Bumped when the on-disk layout changes incompatibly.
FORMAT_VERSION = 2

# Keys that have historically held the parameter tensors.
_STATE_DICT_KEYS = ("state_dict", "model", "weights", "params")

# Prefix `torch.compile` (OptimizedModule) adds to every key.
_COMPILE_PREFIX = "_orig_mod."


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the real module behind a torch.compile wrapper.

    `torch.compile(m).state_dict()` yields `_orig_mod.`-prefixed keys, which
    do not match an eagerly-built GPT. Peeling the wrapper off before saving
    keeps checkpoints portable across compile on/off.
    """
    inner = getattr(model, "_orig_mod", None)
    return inner if isinstance(inner, torch.nn.Module) else model


def strip_compiled_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Remove `_orig_mod.` from keys, for checkpoints saved while compiled."""
    if any(k.startswith(_COMPILE_PREFIX) for k in state_dict):
        return {
            (k[len(_COMPILE_PREFIX):] if k.startswith(_COMPILE_PREFIX) else k): v
            for k, v in state_dict.items()
        }
    return dict(state_dict)


def extract_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the parameter dict out of a checkpoint under any known key."""
    for key in _STATE_DICT_KEYS:
        if key in checkpoint and isinstance(checkpoint[key], Mapping):
            return strip_compiled_prefix(checkpoint[key])
    raise KeyError(
        f"Checkpoint has no parameter dict. Looked for {list(_STATE_DICT_KEYS)}; "
        f"found {sorted(checkpoint.keys())}"
    )


def _torch_load(path: str, map_location: str | torch.device) -> dict[str, Any]:
    """
    torch.load with a safe path first, falling back to the permissive one.

    PyTorch 2.6+ defaults to weights_only=True, which rejects the pickled
    GPTConfig dataclass unless it is allowlisted. We allowlist it, and if that
    still fails (very old torch without safe_globals) we drop the flag.
    """
    try:
        from llm.model import GPTConfig

        with torch.serialization.safe_globals([GPTConfig]):
            return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def save_checkpoint(path: str, model: torch.nn.Module, **extra: Any) -> None:
    """
    Save a model in the canonical format.

    Writes `config` + `state_dict`. Any compiled wrapper is peeled off first,
    and the module is temporarily moved to CPU so a GPU-trained model saves
    portable files.
    """
    model = unwrap(model)

    was_training = model.training
    was_device = next(model.parameters()).device

    try:
        model.to("cpu")
        payload: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "config": model.cfg,  # type: ignore[attr-defined]
            "state_dict": model.state_dict(),
        }
        payload.update(extra)
        torch.save(payload, path)
    finally:
        model.to(was_device)
        model.train(was_training)


def load_checkpoint(
    path: str,
    map_location: str | torch.device = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """
    Load a checkpoint and return `(config, full_checkpoint_dict)`.

    Handles every historical layout: both state-dict key names and
    `_orig_mod.`-prefixed keys from compiled runs.
    """
    checkpoint = _torch_load(path, map_location)

    if "config" not in checkpoint:
        raise KeyError(
            "Checkpoint is missing 'config'. It was probably not written by "
            "this project, or the file is truncated."
        )

    # Normalise onto the canonical key so callers only ever read one name.
    checkpoint["state_dict"] = extract_state_dict(checkpoint)
    return checkpoint["config"], checkpoint