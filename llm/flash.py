"""
flash.py — the fast lane for Shimba LLM models.

Two meanings, one module:

1. The `flash` **arch** (`--arch flash` at train time): RMSNorm, RoPE
   positions, SwiGLU MLP, and grouped-query attention ON by default
   (one kv head per four query heads → ~4× smaller KV cache).
2. **Flash files** (`*_flash.pth`): an existing checkpoint (any arch)
   packed with inference drugs — fp16/int8 weights plus metadata — and
   decoded through the incremental KV-cache path below instead of the
   naive re-read-everything loop in `generate()`.

The cache path lives in `model.py` (`GPT.forward(..., cache=...,
start_pos=...)`); this module is the friendly wrapper around it:
`KVCache` owns the per-layer memories, `fast_generate` runs the
prefill + single-token-step loop, and `pack_flash` / `load_flash`
write and read the `*_flash.pth` artifact.

A flash file loads anywhere a `.pth` does — `GPT.load` ignores the
extra `flash` metadata key — but only the helpers here (or
`flash.py`) actually use the fast path.
"""

from __future__ import annotations

import os
import shutil
from typing import List, Optional

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .generate import _apply_repetition_penalty, _sample_next
from .model import GPT, GPTConfig, cfg_arch, cfg_n_kv
from .tokenizer import CharTokenizer

FLASH_ARCH = "flash"
FLASH_META_KEY = "flash"


# ---------------------------------------------------------------------------
# Arch helpers
# ---------------------------------------------------------------------------

def is_flash_arch(cfg: GPTConfig) -> bool:
    """True when the config was built (or trained) as the flash arch."""
    return cfg_arch(cfg) == FLASH_ARCH


def default_kv_heads(n_head: int) -> int:
    """Flash GQA default: one kv head per four query heads (min 1)."""
    return max(1, n_head // 4)


def describe_flash(cfg: GPTConfig) -> str:
    """One-line summary of why this config is (or isn't) fast."""
    n_kv = cfg_n_kv(cfg)
    ratio = cfg.n_head / max(1, n_kv)
    tag = "flash" if is_flash_arch(cfg) else cfg_arch(cfg)
    return (f"arch={tag} n_head={cfg.n_head} n_head_kv={n_kv} "
            f"(cache {ratio:.0f}x smaller than MHA)")


def cache_bytes_per_token(cfg: GPTConfig) -> int:
    """KV-cache RAM per cached token, in bytes (float32)."""
    n_kv = cfg_n_kv(cfg)
    head_dim = cfg.n_embd // cfg.n_head
    # layers × kv heads × head_dim × (k + v) × 4 bytes
    return cfg.n_layer * n_kv * head_dim * 2 * 4


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

class KVCache:
    """
    Owns the per-layer (k, v) memories for incremental decoding.

    Usage:
        cache = KVCache(model.cfg)
        logits, cache = cache.prefill(model, prompt_ids)  # full prompt, once
        logits, cache = cache.step(model, next_id)        # one token at a time
    """

    def __init__(self, cfg: GPTConfig):
        self.cfg = cfg
        self.caches: Optional[list] = None
        self.n_cached: int = 0

    def reset(self) -> None:
        self.caches = None
        self.n_cached = 0

    def __len__(self) -> int:
        return self.n_cached

    def prefill(self, model: GPT, idx: torch.Tensor):
        """Run the whole prompt once; returns (logits_last, self)."""
        model.eval()
        n_layer = len(model.transformer.h)
        logits, _, caches = model(idx, cache=[None] * n_layer, start_pos=0)
        self.caches = caches
        self.n_cached = idx.size(1)
        return logits, self

    def step(self, model: GPT, token: torch.Tensor):
        """Decode exactly one token; returns (logits_last, self)."""
        assert self.caches is not None, "prefill() before step()"
        model.eval()
        logits, _, caches = model(token, cache=self.caches,
                                  start_pos=self.n_cached)
        self.caches = caches
        self.n_cached += token.size(1)
        return logits, self

    def memory_bytes(self) -> int:
        """Current cache RAM footprint (float32 estimate)."""
        return self.n_cached * cache_bytes_per_token(self.cfg)


# ---------------------------------------------------------------------------
# Fast generation (KV-cache decode loop)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def fast_generate(
    model: GPT,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    stop_tokens: Optional[List[int]] = None,
    stop_strings: Optional[List[str]] = None,
) -> str:
    """
    Generate using the incremental KV-cache path.

    Identical sampling to `generate()` (temperature / top-k / top-p /
    repetition penalty / stop tokens), plus `stop_strings`: generation
    ends when the decoded text ends with one of them (e.g. an
    end-of-answer tag), and the matched suffix is stripped from the
    returned text.

    With temperature=0.0 the output is token-identical to `generate()`.
    """
    model.eval()
    block_size = model.cfg.block_size
    device = next(model.parameters()).device

    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [0]
    if len(ids) > block_size:
        ids = ids[-block_size:]

    stop_set = set(stop_tokens) if stop_tokens else set()
    stop_strs = [s for s in (stop_strings or []) if s]

    cache = KVCache(model.cfg)
    idx = torch.tensor([ids], dtype=torch.int64, device=device)
    logits, cache = cache.prefill(model, idx)
    # The returned text always covers the FULL prompt: `prompt_ids` is never
    # trimmed (only the model's windowed view in `ids` + cache is).
    prompt_ids = list(ids)
    generated_ids: List[int] = []
    seen_ids = set(ids)
    out_text_parts: List[str] = []

    for _ in range(max_new_tokens):
        step_logits = logits[:, -1, :].clone()
        _apply_repetition_penalty(step_logits, seen_ids, repetition_penalty)
        next_id = _sample_next(step_logits, temperature, top_k, top_p)
        token_int = next_id.item()

        if token_int in stop_set:
            break

        generated_ids.append(token_int)
        seen_ids.add(token_int)

        # Sliding window: if the cache is full, drop the oldest entry from
        # both the id list and every layer's (k, v) so positions stay valid.
        # (Rare: only when prompt + output exceeds block_size.)
        if len(ids) + len(generated_ids) > block_size:
            drop = len(ids) + len(generated_ids) - block_size
            ids = ids[drop:]
            trimmed = []
            for (ck, cv) in cache.caches:
                trimmed.append((ck[:, :, drop:, :], cv[:, :, drop:, :]))
            cache.caches = trimmed
            cache.n_cached -= drop

        logits, cache = cache.step(model, next_id)

        piece = tokenizer.decode([token_int])
        out_text_parts.append(piece)
        if stop_strs:
            tail = "".join(out_text_parts)
            hit = next((s for s in stop_strs if tail.endswith(s)), None)
            if hit is not None:
                out_text_parts = ["".join(out_text_parts)[: -len(hit)]]
                break

    return tokenizer.decode(prompt_ids) + "".join(out_text_parts)


# ---------------------------------------------------------------------------
# Flash files: pack / load / info
# ---------------------------------------------------------------------------

def _tokenizer_path_for(model_path: str) -> str:
    return CharTokenizer.default_path(model_path)


def pack_flash(model_path: str, out_path: str,
               dtype: str = "f16") -> dict:
    """
    Pack an inference-ready `*_flash.pth` from a trained checkpoint.

    dtype: "f32" (cache path only, no quality change),
           "f16" (half the file, GPU-fast),
           "int8" (quarter-ish file, CPU-fast; loads back only via
           `load_flash`, like `quantize.py` int8).
    Returns the flash metadata dict written into the file.
    """
    cfg, _ = load_checkpoint(model_path, map_location="cpu")
    model = GPT.load(model_path, device="cpu")
    model.eval()

    drugs = ["kv-cache"]
    if dtype == "f16":
        model = model.half()
        drugs.append("fp16")
    elif dtype == "int8":
        from torch.quantization import quantize_dynamic
        model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        drugs.append("int8")
    elif dtype != "f32":
        raise ValueError(f"dtype must be f32, f16 or int8 (got {dtype})")

    meta = {
        "drugs": drugs,
        "dtype": dtype,
        "arch": cfg_arch(model.cfg),
        "params": sum(p.numel() for p in model.parameters()),
        "kv_cache_per_token_bytes": cache_bytes_per_token(model.cfg),
    }
    save_checkpoint(out_path, model, **{FLASH_META_KEY: meta})

    # The tokenizer is derived from the corpus — the flash file is useless
    # without it, so carry it over next to the output.
    src_tok = _tokenizer_path_for(model_path)
    dst_tok = _tokenizer_path_for(out_path)
    if os.path.abspath(src_tok) != os.path.abspath(dst_tok):
        if not os.path.exists(src_tok):
            raise FileNotFoundError(
                f"Tokenizer not found at '{src_tok}'. "
                f"Pack the original model + tokenizer together.")
        shutil.copy2(src_tok, dst_tok)
    return meta


def read_flash_meta(model_path: str) -> Optional[dict]:
    """The flash metadata dict, or None for a regular checkpoint."""
    _, ckpt = load_checkpoint(model_path, map_location="cpu")
    meta = ckpt.get(FLASH_META_KEY)
    return dict(meta) if isinstance(meta, dict) else None


def is_flash_file(model_path: str) -> bool:
    """True when the file was packed by `pack_flash`."""
    try:
        return read_flash_meta(model_path) is not None
    except Exception:
        return False


def load_flash(model_path: str, device=None) -> tuple:
    """
    Load a `*_flash.pth` (or any regular checkpoint) for fast inference.

    Returns (model, meta) where meta is the flash dict or None. f16
    weights are kept half-precision; the model is set to eval mode.
    """
    meta = read_flash_meta(model_path)
    model = GPT.load(model_path, device=device)
    if meta and meta.get("dtype") == "f16":
        model = model.half()
    model.eval()
    return model, meta


def flash_info(model_path: str) -> str:
    """Human-readable summary of a checkpoint and its flash status."""
    from .bpe import load_tokenizer  # local import: keeps module import light

    cfg, _ = load_checkpoint(model_path, map_location="cpu")
    meta = read_flash_meta(model_path)
    tok_path = _tokenizer_path_for(model_path)
    vocab = (load_tokenizer(tok_path).vocab_size
             if os.path.exists(tok_path) else cfg.vocab_size)
    size_mb = os.path.getsize(model_path) / 1024 ** 2
    n_params = meta.get("params") if meta else None
    if n_params is None:
        n_params = sum(p.numel() for p in GPT.load(model_path,
                                                   device="cpu").parameters())

    lines = [
        f"file    : {model_path} ({size_mb:.1f} MB)",
        f"model   : {describe_flash(cfg)}",
        f"params  : {n_params:,}  vocab {vocab}  block {cfg.block_size}",
        f"tokenizer: {'present' if os.path.exists(tok_path) else 'MISSING → ' + tok_path}",
    ]
    if meta:
        lines.append(f"flash   : drugs={'+'.join(meta.get('drugs', []))} "
                     f"dtype={meta.get('dtype')}")
        lines.append(f"cache/tok: ~{meta.get('kv_cache_per_token_bytes', 0) / 1024:.1f} KB")
    else:
        lines.append("flash   : no — regular checkpoint "
                     "(pack one with flash.py pack)")
    return "\n".join(lines)
