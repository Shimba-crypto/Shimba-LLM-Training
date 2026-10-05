"""
lora.py — Low-Rank Adaptation for Shimba LLM models.

Freezes a trained base model and learns small rank-r side matrices per
Linear layer instead. The base weights never move, so:

  * optimizer state covers ~1% of parameters (12 MB, not 1 GB, at 124M),
  * an adapter checkpoint is a few MB and carries its own config,
  * merging folds the adapters back into plain weights, so the merged
    model runs through every existing path (generate, GGUF, flash)
    with no LoRA-aware code anywhere else.

Math: a frozen W (out x in) gains a delta of (alpha / r) * B @ A, with
A (r x in) ~ N(0, 1/r) and B (out x r) = 0. B starts at zero, so an
injected model is bit-identical to the base until training moves B.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

# Substrings (matched against module names) that receive adapters by
# default: every attention + MLP projection. Embeddings, norms and the
# tied LM head stay frozen — persona and style live in the blocks.
DEFAULT_TARGETS = ("attn.c_attn", "attn.c_proj", "mlp.fc", "mlp.proj",
                   "mlp.gate", "mlp.up")


class LoRALinear(nn.Module):
    """nn.Linear with a trainable low-rank side path; base stays frozen."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        # Share storage with the base weight; never a copy.
        self.weight = base.weight
        self.bias = base.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = nn.functional.linear(x, self.weight, self.bias)
        return out + self.scaling * (x @ self.lora_A.T @ self.lora_B.T)

    def merged_weight(self) -> torch.Tensor:
        """Base weight with the adapter folded in (for export)."""
        return (self.weight.detach()
                + self.scaling * (self.lora_B.detach()
                                  @ self.lora_A.detach()))


def _parent_and_attr(model: nn.Module, dotted: str):
    # ModuleLists (transformer.h.0.attn...) index by int, everything else
    # by attribute name; getattr(module_list, "0") would raise.
    parent = model
    *path, leaf = dotted.split(".")
    for p in path:
        if p.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, leaf


def inject_lora(model: nn.Module, r: int = 8, alpha: float = 16.0,
                targets=DEFAULT_TARGETS, last_layers: int | None = None) -> int:
    """
    Replace matching Linears with LoRALinear in place; freeze everything else.

    last_layers=N restricts adapters to the final N transformer blocks, so
    backprop never reaches the early layers (the main wall-clock win on CPU).
    Returns the trainable parameter count.
    """
    n_blocks = len(model.transformer.h)
    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, LoRALinear):
            continue
        if not any(t in name for t in targets):
            continue
        if last_layers is not None and name.startswith("transformer.h."):
            try:
                idx = int(name.split(".")[2])
            except (ValueError, IndexError):
                idx = -1
            if idx < n_blocks - last_layers:
                continue
        parent, leaf = _parent_and_attr(model, name)
        setattr(parent, leaf, LoRALinear(module, r, alpha))
        count += 1
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)
    return sum(p.numel() for p in lora_parameters(model))


def lora_parameters(model: nn.Module):
    """The only parameters training may touch."""
    return [p for p in model.parameters() if p.requires_grad]


def adapter_state(model: nn.Module) -> dict:
    """Name -> tensor for every LoRA matrix (small enough to save often)."""
    return {n: p.detach().cpu() for n, p in model.named_parameters()
            if p.requires_grad}


def load_adapter_state(model: nn.Module, state: dict) -> None:
    own = dict(model.named_parameters())
    for k, v in state.items():
        if k not in own:
            raise KeyError(f"adapter has unknown parameter: {k}")
        own[k].data.copy_(v)


def merge_lora_(model: nn.Module) -> None:
    """
    Fold every adapter into its base weight in place, then strip the side
    path. After this the model is a plain GPT again: same forward maths as
    training, no LoRA modules left to confuse export or GGUF.
    """
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        module.weight.data.copy_(module.merged_weight())
        plain = nn.Linear(module.weight.shape[1], module.weight.shape[0],
                          bias=module.bias is not None)
        plain.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            plain.bias.data.copy_(module.bias.data)
        parent, leaf = _parent_and_attr(model, name)
        setattr(parent, leaf, plain)
    for p in model.parameters():
        p.requires_grad_(True)


def adapter_info(model: nn.Module) -> str:
    n_train = sum(p.numel() for p in lora_parameters(model))
    n_all = sum(p.numel() for p in model.parameters())
    n_mod = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
    return (f"adapters: {n_mod} linears, {n_train:,} trainable / "
            f"{n_all:,} total ({100 * n_train / max(1, n_all):.2f}%)")
