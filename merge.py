#!/usr/bin/env python3
"""
merge.py — Merge two or more trained Shimba LLM checkpoints into one.

All inputs must share the same architecture (--arch, vocab_size,
block_size, n_embd, n_head, n_layer). This is checked before any math runs.

Methods
-------
average : weighted linear interpolation (works for N models)
    merged = sum(w_i * sd_i) / sum(w_i)

slerp   : spherical linear interpolation (exactly 2 models)
    Interpolates along the sphere per-tensor instead of cutting straight
    through it. Falls back to lerp per-tensor when the two tensors are
    (near-)colinear or one of them is zero.

Usage
-----
    # equal average of two runs
    python merge.py --models a.pth b.pth --out merged.pth

    # weighted average of three runs
    python merge.py --models a.pth b.pth c.pth --weights 0.5 0.3 0.2 --out merged.pth

    # slerp halfway between two runs
    python merge.py --models a.pth b.pth --method slerp --t 0.5 --out merged.pth

    # pick the tokenizer from the second model instead of the first
    python merge.py --models a.pth b.pth --tokenizer-from 1 --out merged.pth

Notes
-----
* The tokenizer JSON sitting beside each model (<stem>_tokenizer.json) must
  describe the same vocabulary, otherwise token ids would silently change
  meaning. Override with --tokenizer-from to pick which one to carry over,
  and --allow-different-tokenizer to force it (not recommended).
* Quantized (int8) checkpoints cannot be merged — merge the fp32 originals.
* Output is written in the canonical checkpoint format, so generate/chat/
  quantize/pth2gguf all accept it directly.
"""

import argparse
import copy
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from llm.checkpoint import FORMAT_VERSION, load_checkpoint
from llm.compat import enable_safe_output
from llm.tokenizer import CharTokenizer

enable_safe_output()

_ARCH_FIELDS = ("vocab_size", "block_size", "n_embd", "n_head", "n_layer")


def _load_state(path: str):
    cfg, ckpt = load_checkpoint(path, map_location="cpu")
    sd = ckpt["state_dict"]
    return cfg, sd


def _arch_of(cfg) -> str:
    return getattr(cfg, "arch", "shimba") or "shimba"


def _check_compatible(cfgs) -> None:
    ref = cfgs[0]
    for i, cfg in enumerate(cfgs[1:], 1):
        for f in _ARCH_FIELDS:
            a, b = getattr(ref, f), getattr(cfg, f)
            if a != b:
                raise ValueError(
                    f"Architecture mismatch: model 0 has {f}={a}, "
                    f"model {i} has {f}={b}. "
                    f"Merge requires identical {list(_ARCH_FIELDS)}."
                )
        # bias affects which keys exist (Linear bias or not)
        if getattr(ref, "bias", False) != getattr(cfg, "bias", False):
            raise ValueError(
                f"Bias mismatch: model 0 bias={getattr(ref, 'bias', False)}, "
                f"model {i} bias={getattr(cfg, 'bias', False)}."
            )
        # arch selects norm/position/MLP structure and the key layout
        if _arch_of(ref) != _arch_of(cfg):
            raise ValueError(
                f"Arch mismatch: model 0 is '{_arch_of(ref)}', "
                f"model {i} is '{_arch_of(cfg)}'. "
                f"Only merge checkpoints trained with the same --arch."
            )
        if (getattr(ref, "n_head_kv", None) or ref.n_head) != \
           (getattr(cfg, "n_head_kv", None) or cfg.n_head):
            raise ValueError(
                f"n_head_kv mismatch between model 0 and model {i}."
            )


def _check_mergeable(sd: dict, path: str) -> None:
    for k, v in sd.items():
        if "causal_mask" in k:
            continue
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"[merge] {path}: key '{k}' is not a tensor")
        if not v.is_floating_point():
            raise ValueError(
                f"[merge] {path}: key '{k}' has dtype {v.dtype} — "
                f"merge the fp32 originals, not quantized exports."
            )


def merge_average(state_dicts, weights) -> dict:
    total = sum(weights)
    if total == 0:
        raise ValueError("--weights must not sum to zero")
    w = [x / total for x in weights]
    keys = state_dicts[0].keys()
    for i, sd in enumerate(state_dicts[1:], 1):
        if set(sd.keys()) != set(keys):
            only_a = sorted(set(keys) - set(sd.keys()))[:5]
            only_b = sorted(set(sd.keys()) - set(keys))[:5]
            raise ValueError(
                f"State-dict keys differ between model 0 and model {i}. "
                f"Only in 0: {only_a}  Only in {i}: {only_b}"
            )
    merged = {}
    for k in keys:
        if "causal_mask" in k:
            merged[k] = state_dicts[0][k]
            continue
        acc = None
        for sd, wi in zip(state_dicts, w):
            t = sd[k].float()
            acc = t * wi if acc is None else acc + t * wi
        merged[k] = acc
    return merged


def _slerp_tensor(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    af, bf = a.float().flatten(), b.float().flatten()
    na, nb = float(af.norm()), float(bf.norm())
    if na == 0.0 or nb == 0.0:
        return (1.0 - t) * a.float() + t * b.float()
    dot = float((af / na * (bf / nb)).sum())
    dot = max(-1.0, min(1.0, dot))
    # Colinear (or opposite) → lerp is the stable choice.
    if abs(dot) > 0.9995:
        return (1.0 - t) * a.float() + t * b.float()
    omega = math.acos(dot)
    so = math.sin(omega)
    if abs(so) < 1e-6:
        return (1.0 - t) * a.float() + t * b.float()
    c0 = math.sin((1.0 - t) * omega) / so
    c1 = math.sin(t * omega) / so
    out = c0 * af + c1 * bf
    return out.reshape(a.shape)


def merge_slerp(state_dicts, t: float) -> dict:
    if len(state_dicts) != 2:
        raise ValueError("--method slerp needs exactly 2 --models")
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"--t must be in [0, 1], got {t}")
    a, b = state_dicts
    if set(a.keys()) != set(b.keys()):
        raise ValueError("State-dict keys differ between the two models.")
    merged = {}
    for k in a.keys():
        if "causal_mask" in k:
            merged[k] = a[k]
            continue
        if a[k].shape != b[k].shape:
            raise ValueError(
                f"Shape mismatch at '{k}': {tuple(a[k].shape)} vs {tuple(b[k].shape)}"
            )
        merged[k] = _slerp_tensor(a[k], b[k], t)
    return merged


def _tokenizer_path(model_path: str) -> str:
    return CharTokenizer.default_path(model_path)


def _load_tokenizer_data(model_path: str) -> dict:
    import json

    tp = _tokenizer_path(model_path)
    if not os.path.exists(tp):
        raise FileNotFoundError(
            f"Tokenizer not found at '{tp}'. "
            f"The tokenizer JSON must sit beside the model file."
        )
    with open(tp, "r", encoding="utf-8") as f:
        return json.load(f)


def _vocab_key(data: dict):
    """Normalized vocab identity for either tokenizer kind."""
    if isinstance(data, dict) and data.get("type") == "bpe":
        return ("bpe", data.get("vocab_size"),
                sorted((k, v) for k, v in data.get("vocab", {}).items()),
                [tuple(m) for m in data.get("merges", [])])
    return ("char", data.get("vocab_size"),
            sorted((k, v) for k, v in (data.get("char2idx") or {}).items()))


def main() -> None:
    p = argparse.ArgumentParser(description="Merge Shimba LLM checkpoints")
    p.add_argument("--models", nargs="+", required=True,
                   help="Two or more input .pth files")
    p.add_argument("--out", required=True, help="Output merged .pth path")
    p.add_argument("--method", choices=["average", "slerp"], default="average")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Per-model weights for --method average "
                        "(default: equal). Must match --models in count.")
    p.add_argument("--t", type=float, default=0.5,
                   help="Interpolation factor for --method slerp (default: 0.5)")
    p.add_argument("--tokenizer-from", type=int, default=0,
                   help="Which input model supplies the output tokenizer (default: 0)")
    p.add_argument("--allow-different-tokenizer", action="store_true",
                   help="Allow merging when tokenizer vocabs differ "
                        "(uses --tokenizer-from; not recommended)")
    args = p.parse_args()

    if len(args.models) < 2:
        print("[error] Need at least 2 --models to merge.")
        sys.exit(1)
    for m in args.models:
        if not os.path.exists(m):
            print(f"[error] Model not found: {m}")
            sys.exit(1)
    if not (0 <= args.tokenizer_from < len(args.models)):
        print(f"[error] --tokenizer-from {args.tokenizer_from} out of range "
              f"for {len(args.models)} models.")
        sys.exit(1)

    if args.method == "average":
        weights = args.weights if args.weights is not None else [1.0] * len(args.models)
        if len(weights) != len(args.models):
            print(f"[error] Got {len(weights)} --weights for "
                  f"{len(args.models)} --models.")
            sys.exit(1)
    else:
        if args.weights is not None:
            print("[warn] --weights is ignored for --method slerp (uses --t).")
        weights = None

    print(f"[merge] loading {len(args.models)} checkpoint(s) ...")
    cfgs, sds = [], []
    for m in args.models:
        cfg, sd = _load_state(m)
        _check_mergeable(sd, m)
        cfgs.append(cfg)
        sds.append(sd)
        n = sum(v.numel() for k, v in sd.items() if "causal_mask" not in k)
        print(f"  {m}: vocab={cfg.vocab_size} block={cfg.block_size} "
              f"n_layer={cfg.n_layer} n_embd={cfg.n_embd} n_head={cfg.n_head} "
              f"params={n:,}")

    try:
        _check_compatible(cfgs)
    except ValueError as e:
        print(f"[error] {e}")
        sys.exit(1)

    # --- tokenizer compatibility --------------------------------------
    try:
        tok_datas = [_load_tokenizer_data(m) for m in args.models]
    except FileNotFoundError as e:
        print(f"[error] {e}")
        sys.exit(1)
    ref_data = tok_datas[args.tokenizer_from]
    ref_key = _vocab_key(ref_data)
    mismatch = [i for i, d in enumerate(tok_datas)
                if _vocab_key(d) != ref_key]
    if mismatch:
        msg = (f"Tokenizers differ between model {args.tokenizer_from} and "
               f"model(s) {mismatch}. Token ids would change meaning.")
        if args.allow_different_tokenizer:
            print(f"[warn] {msg}\n[warn] Continuing with tokenizer from "
                  f"model {args.tokenizer_from} (--allow-different-tokenizer).")
        else:
            print(f"[error] {msg}")
            print("        Merge the tokenizer from one run only, or re-train "
                  "from a shared vocab. Override with "
                  "--allow-different-tokenizer (not recommended).")
            sys.exit(1)
    else:
        print(f"[merge] tokenizers agree "
              f"(vocab_size={ref_data.get('vocab_size')})")

    # --- merge ----------------------------------------------------------
    if args.method == "slerp":
        print(f"[merge] method=slerp t={args.t}")
        merged_sd = merge_slerp(sds, args.t)
        desc = f"slerp(t={args.t}) of {args.models[0]} + {args.models[1]}"
    else:
        assert weights is not None
        total = sum(weights)
        norm = [x / total for x in weights]
        print(f"[merge] method=average weights={[round(x, 4) for x in norm]}")
        merged_sd = merge_average(sds, weights)
        desc = f"average{norm} of {args.models}"

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    out_cfg = copy.deepcopy(cfgs[0])
    payload = {
        "format_version": FORMAT_VERSION,
        "config": out_cfg,
        "state_dict": merged_sd,
        "merge_method": args.method,
        "merge_models": list(args.models),
        "merge_desc": desc,
    }
    torch.save(payload, args.out)
    print(f"[merge] saved → {args.out}")

    # Carry the tokenizer over so the merged model loads immediately.
    # A byte copy preserves both char and BPE files exactly.
    import shutil

    out_tok = CharTokenizer.default_path(args.out)
    shutil.copy2(_tokenizer_path(args.models[args.tokenizer_from]), out_tok)
    print(f"[merge] tokenizer (from model {args.tokenizer_from}) → {out_tok}")

    # Verify the result loads through the normal path.
    from llm.model import GPT

    GPT.load(args.out)
    print("[merge] verified: reload OK")


if __name__ == "__main__":
    main()
