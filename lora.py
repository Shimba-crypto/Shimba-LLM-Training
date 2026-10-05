#!/usr/bin/env python3
"""
lora.py — parameter-efficient fine-tuning for Shimba LLM models.

    python lora.py train --model point-0.2b.pth --data spoon_chat.jsonl \\
        --out spoon-lora --r 8 --last_layers 4
    python lora.py merge --model point-0.2b.pth --adapter spoon-lora \\
        --out spoon-chat.pth

train: freezes the base checkpoint, injects rank-r adapters (optionally
only into the last N blocks, so backprop never touches the early layers),
and saves a few-MB adapter directory: adapter.pt (A/B matrices) +
adapter.json (rank, alpha, targets, base arch). Resume with the same
--out dir and --resume.
merge: folds an adapter back into plain weights and writes a regular .pth
(+ tokenizer copy), loadable everywhere with no LoRA code in the loop.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from llm.checkpoint import load_checkpoint, save_checkpoint
from llm.compat import enable_safe_output
from llm.corpus import load_corpus_text
from llm.data import DataLoader, TextDataset, make_splits
from llm.device import resolve_device
from llm.lora import (DEFAULT_TARGETS, adapter_info, adapter_state,
                      inject_lora, load_adapter_state, merge_lora_)
from llm.model import GPT
from llm.tokenizer import CharTokenizer
from llm.train import TrainConfig, get_lr

enable_safe_output()

ADAPTER_WEIGHTS = "adapter.pt"
ADAPTER_CONFIG = "adapter.json"


def _tokenizer_for(model_path, kind, vocab_size):
    from llm.bpe import BPETokenizer, load_tokenizer
    tok_path = CharTokenizer.default_path(model_path)
    if os.path.exists(tok_path):
        return load_tokenizer(tok_path), tok_path
    raise FileNotFoundError(
        f"Tokenizer not found at '{tok_path}'. LoRA reuses the base "
        f"model's tokenizer; copy it next to {model_path} first.")


def cmd_train(args):
    if not os.path.exists(args.model):
        print(f"[error] Model not found: {args.model}")
        sys.exit(1)
    os.makedirs(args.out, exist_ok=True)
    device = resolve_device(args.device)

    print(f"[lora] base: {args.model}")
    model = GPT.load(args.model, device=device)
    base_cfg = model.cfg
    tok, _ = _tokenizer_for(args.model, args.tokenizer, args.vocab_size)
    if tok.vocab_size != base_cfg.vocab_size:
        print(f"[error] tokenizer vocab {tok.vocab_size} != "
              f"model vocab {base_cfg.vocab_size}")
        sys.exit(1)

    start_iter = 0
    if args.resume:
        cfg_path = os.path.join(args.out, ADAPTER_CONFIG)
        pt_path = os.path.join(args.out, ADAPTER_WEIGHTS)
        if os.path.exists(cfg_path) and os.path.exists(pt_path):
            saved = json.load(open(cfg_path, encoding="utf-8"))
            if saved.get("r") != args.r:
                print(f"[error] saved adapter r={saved.get('r')} != --r {args.r}")
                sys.exit(1)
            n = inject_lora(model, r=args.r,
                            alpha=args.alpha, targets=tuple(args.targets),
                            last_layers=args.last_layers)
            load_adapter_state(model, torch.load(pt_path, map_location="cpu"))
            start_iter = int(saved.get("iter_num", 0))
            print(f"[lora] resumed adapter ({n} params) at iter {start_iter}")
        else:
            print("[lora] --resume given but no adapter found; starting fresh")
            n = inject_lora(model, r=args.r,
                            alpha=args.alpha, targets=tuple(args.targets),
                            last_layers=args.last_layers)
    else:
        n = inject_lora(model, r=args.r,
                        alpha=args.alpha, targets=tuple(args.targets),
                        last_layers=args.last_layers)
    print(f"[lora] {adapter_info(model)}")

    print(f"[lora] corpus: {args.data}")
    text = load_corpus_text(args.data)
    tokens = tok.encode(text)
    del text
    train_ds, val_ds = make_splits(tokens, base_cfg.block_size,
                                   val_fraction=args.val_frac)
    del tokens
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False)

    sched = TrainConfig(learning_rate=args.lr, max_iters=args.max_iters,
                        warmup_iters=args.warmup, min_lr=args.min_lr)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, 0.95))
    model.train()
    model.to(device)

    def estimate():
        model.eval()
        out = {}
        with torch.inference_mode():
            for name, loader in (("train", train_loader), ("val", val_loader)):
                tot, cnt = 0.0, 0
                for _ in range(args.eval_iters):
                    x, y = loader.get_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.amp.autocast("cuda", enabled=args.amp):
                        _, loss = model(x, y)
                    tot += loss.item()
                    cnt += 1
                out[name] = tot / max(1, cnt)
        model.train()
        return out

    def save(iter_num, best_val):
        torch.save(adapter_state(model),
                   os.path.join(args.out, ADAPTER_WEIGHTS))
        json.dump({
            "r": args.r, "alpha": args.alpha, "targets": list(args.targets),
            "last_layers": args.last_layers, "base": args.model,
            "arch": getattr(base_cfg, "arch", "shimba"),
            "iter_num": iter_num, "best_val_loss": best_val,
        }, open(os.path.join(args.out, ADAPTER_CONFIG), "w",
                encoding="utf-8"), indent=1)

    t0 = time.time()
    best_val = float("inf")
    if start_iter:
        cfg_path = os.path.join(args.out, ADAPTER_CONFIG)
        best_val = json.load(open(cfg_path, encoding="utf-8")).get(
            "best_val_loss", best_val)
    try:
        for it in range(start_iter, args.max_iters):
            lr = get_lr(sched, it)
            for g in opt.param_groups:
                g["lr"] = lr
            if it % args.eval_interval == 0:
                losses = estimate()
                el = time.time() - t0
                print(f"  iter {it:6d}/{args.max_iters}  "
                      f"train={losses['train']:.4f}  val={losses['val']:.4f}  "
                      f"lr={lr:.2e}  {el:.0f}s")
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    save(it, best_val)
                    print(f"  ✓ new best (val_loss={best_val:.4f})")
            x, y = train_loader.get_batch()
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            if (it + 1) % args.save_every == 0:
                save(it + 1, best_val)
    except KeyboardInterrupt:
        print("\n[lora] interrupted — saving adapter ...")
        save(it if "it" in dir() else start_iter, best_val)
        sys.exit(0)

    save(args.max_iters, best_val)
    print(f"\n[lora] done  best val_loss={best_val:.4f}  adapter → {args.out}/")


def cmd_merge(args):
    if not os.path.exists(args.model):
        print(f"[error] Model not found: {args.model}")
        sys.exit(1)
    cfg_path = os.path.join(args.adapter, ADAPTER_CONFIG)
    pt_path = os.path.join(args.adapter, ADAPTER_WEIGHTS)
    for p in (cfg_path, pt_path):
        if not os.path.exists(p):
            print(f"[error] Adapter file not found: {p}")
            sys.exit(1)
    saved = json.load(open(cfg_path, encoding="utf-8"))
    device = resolve_device(args.device)

    model = GPT.load(args.model, device=device)
    inject_lora(model, r=saved["r"], alpha=saved.get("alpha", 16.0),
                targets=tuple(saved.get("targets", list(DEFAULT_TARGETS))),
                last_layers=saved.get("last_layers"))
    load_adapter_state(model, torch.load(pt_path, map_location="cpu"))
    print(f"[lora] merging adapter ({saved['r']=}, "
          f"base iters={saved.get('iter_num')}) ...")
    merge_lora_(model)
    model.eval()
    save_checkpoint(args.out, model, lora_merged_from=args.adapter)
    src_tok = CharTokenizer.default_path(args.model)
    dst_tok = CharTokenizer.default_path(args.out)
    if os.path.abspath(src_tok) != os.path.abspath(dst_tok):
        import shutil
        shutil.copy2(src_tok, dst_tok)
    size_mb = os.path.getsize(args.out) / 1024 ** 2
    print(f"[lora] merged → {args.out} ({size_mb:.1f} MB) + tokenizer")


def main():
    p = argparse.ArgumentParser(description="Shimba LLM — LoRA fine-tuning")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train a LoRA adapter on frozen base")
    t.add_argument("--model", required=True, help="Frozen base .pth")
    t.add_argument("--data", required=True, help="Corpus file or folder")
    t.add_argument("--out", required=True, help="Adapter directory")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--r", type=int, default=8)
    t.add_argument("--alpha", type=float, default=16.0)
    t.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    t.add_argument("--last_layers", type=int, default=None,
                   help="Adapters only in final N blocks (CPU speed)")
    t.add_argument("--tokenizer", default="bpe", choices=["char", "bpe"],
                   help="Must match the base model's kind")
    t.add_argument("--vocab-size", type=int, default=2000)
    t.add_argument("--val_frac", type=float, default=0.1)
    t.add_argument("--batch_size", type=int, default=8)
    t.add_argument("--max_iters", type=int, default=300)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--min_lr", type=float, default=1e-5)
    t.add_argument("--warmup", type=int, default=25)
    t.add_argument("--eval_interval", type=int, default=50)
    t.add_argument("--eval_iters", type=int, default=10)
    t.add_argument("--save_every", type=int, default=50)
    t.add_argument("--device", default="auto")
    t.add_argument("--amp", action="store_true")
    t.set_defaults(func=cmd_train)

    m = sub.add_parser("merge", help="Fold adapter into plain weights")
    m.add_argument("--model", required=True, help="Frozen base .pth")
    m.add_argument("--adapter", required=True, help="Adapter directory")
    m.add_argument("--out", required=True, help="Merged output .pth")
    m.add_argument("--device", default="auto")
    m.set_defaults(func=cmd_merge)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
