#!/usr/bin/env python3
"""
quick_train.py — Training with a checkpoint saved every iteration.

Written for Colab: the runtime can disconnect at any moment, so state is
flushed to disk after every optimiser step and `--resume` picks up exactly
where it left off. I/O is throttled to avoid spending more time writing
checkpoints than training.

Usage:
    python quick_train.py --data corpus.txt --out model.pth
    python quick_train.py --data corpus.txt --out model.pth --resume
    python quick_train.py --data ./texts --out runs/shimba.pth --device cuda --amp
    python quick_train.py --data data.jsonl --out runs/shimba.pth
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from llm.checkpoint import save_checkpoint
from llm.compat import enable_safe_output
from llm.data import make_splits
from llm.device import describe_device
from llm.model import GPTConfig
from llm.tokenizer import CharTokenizer
from llm.train import Trainer, TrainConfig, get_lr, resume_from

enable_safe_output()


def load_corpus(path: str, pattern: str = "*.txt,*.json,*.jsonl",
                text_field=None, data_format: str = "auto") -> str:
    """Read a single file, or merge every matching file under a directory."""
    from llm.corpus import load_corpus_text

    return load_corpus_text(
        path,
        pattern=pattern,
        recurse=True,
        text_field=text_field,
        data_format=data_format,
    )


def main():
    p = argparse.ArgumentParser(description="Shimba LLM — checkpoint-every-step trainer")
    p.add_argument("--data", required=True, help="Path to a .txt/.json/.jsonl file OR a folder of them")
    p.add_argument("--out", default="model.pth")
    p.add_argument("--resume", action="store_true", help="Continue from --out if it exists")
    p.add_argument("--pattern", default="*.txt,*.json,*.jsonl",
                   help="Comma-separated glob(s) when --data is a folder")
    p.add_argument("--text_field", default=None,
                   help="JSON field(s) to train on, comma-separated (default: auto-detect)")
    p.add_argument("--format", default="auto", choices=["auto", "txt", "json", "jsonl"],
                   help="Corpus parser (default: auto-detect from extension)")
    p.add_argument("--tokenizer", default="char", choices=["char", "bpe"],
                   help="Tokenizer: char (default) or byte-level BPE")
    p.add_argument("--vocab-size", type=int, default=2000,
                   help="BPE vocabulary size target (default: 2000)")
    # Model
    p.add_argument("--arch", default="shimba", choices=["shimba", "gpt2", "llama"],
                   help="Block architecture (default: shimba)")
    p.add_argument("--n_embd", type=int, default=64)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--eval_iters", type=int, default=20)
    p.add_argument("--val_frac", type=float, default=0.1)
    # Runtime
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N | mps")
    p.add_argument("--amp", action="store_true", help="Mixed precision on CUDA")
    p.add_argument("--compile", action="store_true", help="Enable torch.compile")
    p.add_argument("--save_every", type=int, default=1,
                   help="Write a checkpoint every N iterations (default: 1)")
    args = p.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    tok_path = CharTokenizer.default_path(args.out)

    # --- tokenizer: reuse on resume so token ids stay stable -------------
    resuming = args.resume and os.path.exists(args.out) and os.path.exists(tok_path)

    print(f"[quick_train] loading corpus from {args.data} ...")
    text = load_corpus(args.data, args.pattern, args.text_field, args.format)

    if resuming:
        from llm.bpe import load_tokenizer
        print("[quick_train] resume: loading existing tokenizer ...")
        tokenizer = load_tokenizer(tok_path)
        saved_kind = getattr(tokenizer, "TYPE", "char")
        if saved_kind != args.tokenizer:
            print(f"[quick_train] saved tokenizer is '{saved_kind}' but "
                  f"--tokenizer is '{args.tokenizer}'. "
                  f"Resume with the matching kind.")
            sys.exit(1)
    elif args.tokenizer == "bpe":
        from llm.bpe import BPETokenizer
        print("[quick_train] training BPE tokenizer ...")
        tokenizer = BPETokenizer().build(text, vocab_size=args.vocab_size)
        tokenizer.save(tok_path)
    else:
        print("[quick_train] building tokenizer ...")
        tokenizer = CharTokenizer().build(text)
        tokenizer.save(tok_path)

    if len(text) < args.block_size * 2:
        print(f"[quick_train] corpus too short ({len(text):,} chars), "
              f"need at least {args.block_size * 2:,}")
        sys.exit(1)

    print("[quick_train] encoding corpus ...")
    tokens = tokenizer.encode(text)
    print(f"[quick_train] total tokens: {len(tokens):,}")
    del text

    train_ds, val_ds = make_splits(tokens, args.block_size, val_fraction=args.val_frac)
    del tokens

    # --- architecture: inherit from the checkpoint when resuming --------
    if resuming:
        from llm.checkpoint import load_checkpoint

        try:
            model_cfg = load_checkpoint(args.out)[0]
        except Exception as e:
            print(f"[error] checkpoint at '{args.out}' is unreadable ({e}).")
            print("        It was likely truncated by a kill during saving "
                  "(versions before atomic saves). Delete the .pth to start "
                  "over — the tokenizer rebuilds once, then training runs.")
            sys.exit(1)
        print(f"[quick_train] resume: saved architecture "
              f"(arch={getattr(model_cfg, 'arch', 'shimba')}, "
              f"vocab={model_cfg.vocab_size}, n_layer={model_cfg.n_layer}, "
              f"n_embd={model_cfg.n_embd}, block_size={model_cfg.block_size})")
        if model_cfg.vocab_size != tokenizer.vocab_size:
            print(f"[quick_train] vocab mismatch: model wants "
                  f"{model_cfg.vocab_size}, tokenizer has {tokenizer.vocab_size}")
            sys.exit(1)
        if getattr(model_cfg, "arch", "shimba") != args.arch:
            print(f"[quick_train] arch mismatch: checkpoint is "
                  f"'{getattr(model_cfg, 'arch', 'shimba')}' but --arch is "
                  f"'{args.arch}'. Resume with the matching --arch.")
            sys.exit(1)
    else:
        model_cfg = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            dropout=args.dropout,
            bias=args.arch == "gpt2",
            arch=args.arch,
        )

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_iters=args.max_iters,
        weight_decay=args.weight_decay,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        log_interval=max(1, args.eval_interval // 4 or 1),
        out_path=args.out,
        device=args.device,
        amp=args.amp,
        compile=args.compile,
    )

    trainer = Trainer(model_cfg, train_cfg, train_ds, val_ds)

    if resuming:
        resume_from(trainer, args.out, args.max_iters)

    n_params = sum(q.numel() for q in trainer.model.parameters())
    print(f"[quick_train] device={describe_device(trainer.device)}  params={n_params:,}")
    print(f"[quick_train] checkpointing every {args.save_every} iter → {args.out}")
    print("[quick_train] Ctrl+C is safe: state is already on disk. "
          "Resume with --resume\n")

    t0 = time.time()
    trainer.optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    accum_count = 0
    last_save = 0.0

    try:
        while trainer.iter_num < train_cfg.max_iters:
            lr = get_lr(train_cfg, trainer.iter_num)
            for group in trainer.optimizer.param_groups:
                group["lr"] = lr

            if trainer.iter_num % train_cfg.eval_interval == 0:
                losses = trainer._estimate_loss()
                elapsed = time.time() - t0
                done = trainer.iter_num / train_cfg.max_iters * 100
                print(f"  iter {trainer.iter_num:6d}/{train_cfg.max_iters}  "
                      f"train={losses['train']:.4f}  val={losses['val']:.4f}  "
                      f"lr={lr:.2e}  {elapsed:.0f}s  ({done:.0f}%)")
                if losses["val"] < trainer.best_val_loss:
                    trainer.best_val_loss = losses["val"]
                    trainer._save(args.out)
                    print(f"  ✓ new best (val_loss={trainer.best_val_loss:.4f})")

            for _ in range(train_cfg.gradient_accumulation_steps):
                x, y = trainer.train_loader.get_batch()
                x, y = x.to(trainer.device), y.to(trainer.device)
                _, loss = trainer._forward(x, y)
                loss = loss / train_cfg.gradient_accumulation_steps
                trainer.scaler.scale(loss).backward()
                accum_loss += loss.item()
                accum_count += 1

            trainer.scaler.unscale_(trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), train_cfg.grad_clip)
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
            trainer.optimizer.zero_grad(set_to_none=True)

            trainer.iter_num += 1

            # Throttled checkpoint: every N iters, or at least every 60s.
            now = time.time()
            due = (trainer.iter_num % args.save_every == 0) or (now - last_save >= 60)
            if due:
                trainer._save(args.out)
                last_save = now

            if trainer.iter_num % train_cfg.log_interval == 0 and trainer.iter_num > 0:
                # Average, not sum — accumulating raw per-step losses makes the
                # number scale with log_interval and reads as nonsense.
                avg = accum_loss / max(1, accum_count)
                print(f"  iter {trainer.iter_num:6d}  loss={avg:.4f}  lr={lr:.2e}")
                accum_loss = 0.0
                accum_count = 0

    except KeyboardInterrupt:
        print("\n[quick_train] interrupted — saving state ...")
        trainer._save(args.out)
        print(f"[quick_train] saved at iteration {trainer.iter_num}. "
              f"Resume with --resume")
        sys.exit(0)

    trainer._save(args.out)
    print(f"\n[quick_train] done in {time.time() - t0:.0f}s  "
          f"best val_loss={trainer.best_val_loss:.4f}")
    print(f"[quick_train] model    → {args.out}")
    print(f"[quick_train] tokenizer → {tok_path}")


if __name__ == "__main__":
    main()