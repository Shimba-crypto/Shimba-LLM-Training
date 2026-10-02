#!/usr/bin/env python3
"""
setup.py — Shimba LLM entry point.

Usage
-----
  python setup.py train    --data <file.txt>        --out <model.pth> [options]
  python setup.py train    --data <folder/>          --out <model.pth> [options]
  python setup.py generate --prompt "..." --model <model.pth> [options]
  python setup.py help

When --data points to a FOLDER, all .txt files inside it are merged
(recursively) into one corpus before training.

Train options
-------------
  --data          Path to a .txt file OR a folder of .txt files (required)
  --out           Output model path (default: model.pth)
  --block_size    Context window length          (default: 512)
  --n_embd        Embedding dimension            (default: 256)
  --n_head        Number of attention heads      (default: 4)
  --n_layer       Number of transformer layers   (default: 4)
  --dropout       Dropout probability            (default: 0.1)
  --batch_size    Mini-batch size                (default: 8)
  --grad_accum    Gradient accumulation steps    (default: 4)
  --lr            Learning rate                  (default: 3e-4)
  --max_iters     Training iterations            (default: 5000)
  --eval_interval Eval every N iters             (default: 500)
  --eval_iters    Batches per eval               (default: 100)
  --weight_decay  AdamW weight decay             (default: 0.1)
  --val_frac      Fraction of data for validation (default: 0.1)
  --pattern       Glob pattern when --data is a folder (default: *.txt)
  --no_recurse    Do not recurse into subfolders (flag)
  --device        auto | cpu | cuda | cuda:N | mps   (default: auto)
  --amp           Mixed precision on CUDA (flag)
  --compile       Enable torch.compile (slow first run, flag)
  --resume        Continue from an existing checkpoint at --out (flag)

Generate options
----------------
  --prompt        Seed text for generation (required)
  --model         Path to saved model .pth file (required)
  --max_tokens    Tokens to generate        (default: 200)
  --temperature   Sampling temperature      (default: 0.8)
  --top_k         Top-k sampling            (default: 40)
  --top_p         Nucleus sampling p        (default: 0.95)
  --stream        Stream output token-by-token (flag)

Notes
-----
  * The tokenizer is saved alongside the model as <model_stem>_tokenizer.json
  * Runs on CUDA / MPS / CPU — auto-detected by default, so it works on Colab
    (GPU), Apple Silicon, and a plain laptop with no changes.
  * Checkpoints are always written as CPU tensors, so a model trained on a
    Colab GPU loads anywhere.
"""

import sys
import os
import argparse
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.compat import enable_safe_output

enable_safe_output()


# ---------------------------------------------------------------------------
# Folder → merged corpus loader
# ---------------------------------------------------------------------------

def load_data_path(data_path: str, pattern: str = "*.txt", recurse: bool = True) -> str:
    """
    Load text from:
      - a single .txt file, OR
      - a directory: merges all files matching `pattern` (optionally recursive)

    Returns the combined text string.
    """
    if os.path.isfile(data_path):
        return _read_file(data_path)

    if os.path.isdir(data_path):
        # Collect matching files
        if recurse:
            matches = []
            for root, dirs, files in os.walk(data_path):
                # Sort for deterministic ordering
                dirs.sort()
                for fname in sorted(files):
                    if _matches_pattern(fname, pattern):
                        matches.append(os.path.join(root, fname))
        else:
            matches = sorted(glob.glob(os.path.join(data_path, pattern)))

        if not matches:
            print(f"[error] No files matching '{pattern}' found in '{data_path}'")
            sys.exit(1)

        print(f"[data] found {len(matches)} file(s) in '{data_path}'")

        # Read and concatenate with a separator so docs don't bleed together
        sep = "\n\n" + "=" * 60 + "\n\n"
        parts = []
        total_bytes = 0
        for i, fpath in enumerate(matches, 1):
            text = _read_file(fpath, silent=True)
            if text.strip():          # skip empty files
                parts.append(text)
                total_bytes += len(text)
            # Progress every 20 files
            if i % 20 == 0 or i == len(matches):
                print(f"  loaded {i}/{len(matches)} files  "
                      f"({total_bytes / 1_000_000:.1f} MB so far)")

        combined = sep.join(parts)
        print(f"[data] total corpus: {len(combined):,} characters "
              f"across {len(parts)} file(s)")
        return combined

    print(f"[error] --data path does not exist: '{data_path}'")
    sys.exit(1)


def _matches_pattern(filename: str, pattern: str) -> bool:
    """Simple glob-style match on filename only (not full path)."""
    import fnmatch
    return fnmatch.fnmatch(filename.lower(), pattern.lower())


def _read_file(path: str, silent: bool = False) -> str:
    """Read a text file with UTF-8 → latin-1 fallback."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            text = f.read()
    if not silent:
        print(f"[data] loaded '{path}'  ({len(text):,} characters)")
    return text


# ---------------------------------------------------------------------------
# Sub-command: train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    from llm.model import GPTConfig
    from llm.tokenizer import CharTokenizer
    from llm.data import make_splits
    from llm.train import Trainer, TrainConfig, resume_from
    from llm.checkpoint import load_checkpoint

    # 1. Load data (file or folder)
    text = load_data_path(args.data, pattern=args.pattern, recurse=not args.no_recurse)

    if len(text) < args.block_size * 2:
        print(f"[error] Corpus too short ({len(text):,} chars). "
              f"Need at least {args.block_size * 2:,} characters.")
        sys.exit(1)

    # Warn if corpus is very large (memory check)
    size_mb = len(text) / 1_000_000
    if size_mb > 500:
        print(f"[warn] Large corpus ({size_mb:.0f} MB). "
              f"Token tensor will use ~{size_mb * 2:.0f} MB RAM (int16). "
              f"Consider --block_size 256 or --batch_size 4 to save memory.")

    # Make sure output directory exists before anything writes there
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    tok_path = CharTokenizer.default_path(args.out)

    # 2. Tokenizer — on resume, reuse the saved one so token ids stay stable.
    #    Rebuilding from the corpus would reshuffle the vocab and invalidate
    #    every embedding already learned.
    resuming = args.resume and os.path.exists(args.out) and os.path.exists(tok_path)

    if resuming:
        print("[train] resume: loading existing tokenizer ...")
        tokenizer = CharTokenizer.load(tok_path)
    else:
        print("[train] building tokenizer ...")
        tokenizer = CharTokenizer().build(text)
        tokenizer.save(tok_path)

    # 3. Encode corpus
    print("[train] encoding corpus (this may take a moment for large files) ...")
    tokens = tokenizer.encode(text)
    print(f"[train] total tokens: {len(tokens):,}")

    # Free the raw text string — we only need tokens from here on
    del text

    # 4. Split into train / val
    train_ds, val_ds = make_splits(tokens, args.block_size, val_fraction=args.val_frac)
    del tokens   # free encoded list; datasets hold int16 tensors

    # 5. Model config — on resume, inherit the checkpoint's architecture.
    #    A vocab_size mismatch against the saved tokenizer would throw deep
    #    inside the embedding lookup, so surface it here instead.
    if resuming:
        model_cfg = load_checkpoint(args.out)[0]
        print(f"[train] resume: using saved architecture "
              f"(vocab={model_cfg.vocab_size}, n_layer={model_cfg.n_layer}, "
              f"n_embd={model_cfg.n_embd}, block_size={model_cfg.block_size})")
        if model_cfg.vocab_size != tokenizer.vocab_size:
            print(f"[error] Saved model expects vocab_size={model_cfg.vocab_size} "
                  f"but the tokenizer has {tokenizer.vocab_size}.")
            print("        The tokenizer and checkpoint are out of sync — "
                  "start training from scratch instead.")
            sys.exit(1)
        if model_cfg.block_size != args.block_size:
            print(f"[warn] --block_size {args.block_size} differs from the "
                  f"saved {model_cfg.block_size}; using the saved value.")
    else:
        model_cfg = GPTConfig(
            vocab_size = tokenizer.vocab_size,
            block_size = args.block_size,
            n_embd     = args.n_embd,
            n_head     = args.n_head,
            n_layer    = args.n_layer,
            dropout    = args.dropout,
            bias       = False,
        )

    # 6. Training config
    train_cfg = TrainConfig(
        batch_size                  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        max_iters                   = args.max_iters,
        weight_decay                = args.weight_decay,
        eval_interval               = args.eval_interval,
        eval_iters                  = args.eval_iters,
        out_path                    = args.out,
        device                      = args.device,
        amp                         = args.amp,
        compile                     = args.compile,
    )

    # 7. Train
    trainer = Trainer(model_cfg, train_cfg, train_ds, val_ds)

    if resuming:
        resume_from(trainer, args.out, args.max_iters)

    trainer.run()


# ---------------------------------------------------------------------------
# Sub-command: generate
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> None:
    from llm.model import GPT
    from llm.tokenizer import CharTokenizer
    from llm.generate import generate, stream_generate

    # Check the model exists *before* touching the tokenizer — otherwise a bad
    # --model path reports a confusing "tokenizer not found" error instead.
    if not os.path.exists(args.model):
        print(f"[error] Model file not found: '{args.model}'")
        sys.exit(1)

    tok_path = CharTokenizer.default_path(args.model)
    if not os.path.exists(tok_path):
        print(f"[error] Tokenizer not found at '{tok_path}'.")
        print("        The tokenizer JSON must sit beside the model file.")
        sys.exit(1)
    tokenizer = CharTokenizer.load(tok_path)

    model = GPT.load(args.model, device=args.device)

    print(f"\n[generate] prompt     : {args.prompt!r}")
    print(f"[generate] temperature: {args.temperature}  "
          f"top_k: {args.top_k}  top_p: {args.top_p}  "
          f"max_tokens: {args.max_tokens}")
    print("\n" + "-" * 60)

    if args.stream:
        stream_generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    else:
        result = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(result)

    print("-" * 60)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python setup.py",
        description="Shimba LLM — tiny decoder-only transformer (CPU / CUDA / MPS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── train ────────────────────────────────────────────────────────
    t = sub.add_parser("train", help="Train a new model on text file(s)")
    t.add_argument("--data",          required=True,
                   help="Path to a .txt file OR a folder containing .txt files")
    t.add_argument("--out",           default="model.pth",
                   help="Output model path (default: model.pth)")
    t.add_argument("--pattern",       default="*.txt",
                   help="File glob pattern when --data is a folder (default: *.txt)")
    t.add_argument("--no_recurse",    action="store_true",
                   help="Do not search subfolders (top-level only)")
    # Model
    t.add_argument("--block_size",    type=int,   default=512)
    t.add_argument("--n_embd",        type=int,   default=256)
    t.add_argument("--n_head",        type=int,   default=4)
    t.add_argument("--n_layer",       type=int,   default=4)
    t.add_argument("--dropout",       type=float, default=0.1)
    # Training
    t.add_argument("--batch_size",    type=int,   default=8)
    t.add_argument("--grad_accum",    type=int,   default=4)
    t.add_argument("--lr",            type=float, default=3e-4)
    t.add_argument("--max_iters",     type=int,   default=5000)
    t.add_argument("--weight_decay",  type=float, default=0.1)
    t.add_argument("--eval_interval", type=int,   default=500)
    t.add_argument("--eval_iters",    type=int,   default=100)
    t.add_argument("--val_frac",      type=float, default=0.1)
    # Runtime
    t.add_argument("--device",        default="auto",
                   help="auto | cpu | cuda | cuda:N | mps (default: auto)")
    t.add_argument("--amp",           action="store_true",
                   help="Mixed precision on CUDA (bf16/fp16)")
    t.add_argument("--compile",       action="store_true",
                   help="Enable torch.compile (first run is slow)")
    t.add_argument("--resume",        action="store_true",
                   help="Continue training from an existing checkpoint at --out")

    # ── generate ─────────────────────────────────────────────────────
    g = sub.add_parser("generate", help="Generate text from a trained model")
    g.add_argument("--prompt",      required=True)
    g.add_argument("--model",       required=True)
    g.add_argument("--max_tokens",  type=int,   default=200)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top_k",       type=int,   default=40)
    g.add_argument("--top_p",       type=float, default=0.95)
    g.add_argument("--stream",      action="store_true")
    g.add_argument("--device",      default="auto",
                   help="auto | cpu | cuda | cuda:N | mps (default: auto)")

    # ── help ─────────────────────────────────────────────────────────
    sub.add_parser("help", help="Show detailed help")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        print("\nExamples:")
        print("  # single file")
        print("  python setup.py train --data corpus.txt --out model.pth")
        print()
        print("  # entire folder of .txt files")
        print("  python setup.py train --data ./texts --out model.pth")
        print()
        print("  # Colab GPU, mixed precision")
        print("  python setup.py train --data corpus.txt --out model.pth --amp")
        print()
        print("  # resume an interrupted run")
        print("  python setup.py train --data corpus.txt --out model.pth --resume")
        print()
        print("  # generate")
        print('  python setup.py generate --model model.pth --prompt "This agreement"')
        sys.exit(0)

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "help":
        parser.print_help()
        print("\n" + __doc__)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()