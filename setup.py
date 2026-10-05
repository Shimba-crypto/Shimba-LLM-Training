#!/usr/bin/env python3
"""
setup.py — Shimba LLM entry point.

Usage
-----
  python setup.py train    --data <file.txt>        --out <model.pth> [options]
  python setup.py train    --data <folder/>          --out <model.pth> [options]
  python setup.py train    --data data.jsonl         --out <model.pth> [options]
  python setup.py generate --prompt "..." --model <model.pth> [options]
  python setup.py merge    --models a.pth b.pth --out merged.pth [options]
  python setup.py gguf     --model model.pth --out model.gguf [options]
  python setup.py help

When --data points to a FOLDER, all matching files inside it are merged
(recursively) into one corpus before training. Each file is parsed by its
extension: .txt as raw text, .json as a JSON array/object, .jsonl as one
JSON value per line (non-JSON lines are kept as plain text).

Train options
-------------
  --data          Path to a .txt/.json/.jsonl file OR a folder of them (required)
  --out           Output model path (default: model.pth)
  --arch          shimba | gpt2 | llama | flash (default: shimba)
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
  --text_field    JSON field(s) to train on, comma-separated (default: auto-detect)
  --format        Corpus parser: auto | txt | json | jsonl (default: auto)
  --pattern       Glob pattern(s) when --data is a folder (default: *.txt,*.json,*.jsonl)
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

Merge options
-------------
  --models        Two or more input .pth files (required)
  --out           Output merged .pth path (required)
  --method        average | slerp (default: average)
  --weights       Per-model weights for average (default: equal)
  --t             Interpolation factor for slerp (default: 0.5)
  --tokenizer-from  Which input supplies the tokenizer (default: 0)

GGUF options
------------
  --model         Input model .pth file (required)
  --out           Output .gguf path (required)
  --outtype       f32 | f16 | q8_0 (default: f32)
  --compat        auto | none | gpt2 | llama (default: auto follows checkpoint)
  --name          Model name for metadata (default: input stem)

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
# Folder → merged corpus loader (.txt / .json / .jsonl)
# ---------------------------------------------------------------------------

def load_data_path(data_path: str, pattern: str = "*.txt,*.json,*.jsonl",
                   recurse: bool = True, text_field=None,
                   data_format: str = "auto") -> str:
    """
    Load text from:
      - a single .txt / .json / .jsonl file, OR
      - a directory: merges all files matching `pattern` (optionally recursive)

    `pattern` accepts comma-separated globs (default covers all three
    extensions). `text_field` names JSON object field(s) to read first.
    `data_format` forces one parser ("auto" detects from the extension).

    Returns the combined text string.
    """
    from llm.corpus import load_corpus_text

    return load_corpus_text(
        data_path,
        pattern=pattern,
        recurse=recurse,
        text_field=text_field,
        data_format=data_format,
    )


def _matches_pattern(filename: str, pattern: str) -> bool:
    """Glob-style match on filename only; `pattern` may be comma-separated."""
    from llm.corpus import matches_any, split_patterns

    return matches_any(filename, split_patterns(pattern))


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

    # 1. Load data (file or folder; .txt / .json / .jsonl)
    text = load_data_path(args.data, pattern=args.pattern, recurse=not args.no_recurse,
                          text_field=args.text_field, data_format=args.format)

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
        from llm.bpe import load_tokenizer
        tokenizer = load_tokenizer(tok_path)
        saved_kind = getattr(tokenizer, "TYPE", "char")
        if saved_kind != args.tokenizer:
            print(f"[error] Saved tokenizer is '{saved_kind}' but --tokenizer "
                  f"is '{args.tokenizer}'. Resume with the matching kind.")
            sys.exit(1)
    elif args.tokenizer == "bpe":
        from llm.bpe import BPETokenizer
        print("[train] training BPE tokenizer ...")
        tokenizer = BPETokenizer().build(text, vocab_size=args.vocab_size)
        tokenizer.save(tok_path)
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
        try:
            model_cfg = load_checkpoint(args.out)[0]
        except Exception as e:
            print(f"[error] checkpoint at '{args.out}' is unreadable ({e}).")
            print("        Delete the .pth to start over (the tokenizer "
                  "rebuilds once, then training runs).")
            sys.exit(1)
        print(f"[train] resume: using saved architecture "
              f"(arch={getattr(model_cfg, 'arch', 'shimba')}, "
              f"vocab={model_cfg.vocab_size}, n_layer={model_cfg.n_layer}, "
              f"n_embd={model_cfg.n_embd}, block_size={model_cfg.block_size})")
        if model_cfg.vocab_size != tokenizer.vocab_size:
            print(f"[error] Saved model expects vocab_size={model_cfg.vocab_size} "
                  f"but the tokenizer has {tokenizer.vocab_size}.")
            print("        The tokenizer and checkpoint are out of sync — "
                  "start training from scratch instead.")
            sys.exit(1)
        if getattr(model_cfg, "arch", "shimba") != args.arch:
            print(f"[error] Saved model is arch "
                  f"'{getattr(model_cfg, 'arch', 'shimba')}' but --arch is "
                  f"'{args.arch}'. Resume with the matching --arch.")
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
            bias       = args.arch == "gpt2",
            arch       = args.arch,
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

    from llm.scw import load_tokenizer_for
    from llm.tokenizer import CharTokenizer
    if args.model.endswith(".scw") or os.path.exists(
            CharTokenizer.default_path(args.model)):
        try:
            tokenizer = load_tokenizer_for(args.model)
        except (FileNotFoundError, ValueError) as e:
            print(f"[error] {e}")
            sys.exit(1)
    else:
        tok_path = CharTokenizer.default_path(args.model)
        print(f"[error] Tokenizer not found at '{tok_path}'.")
        print("        The tokenizer JSON must sit beside the model file.")
        sys.exit(1)

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
# Sub-command: merge
# ---------------------------------------------------------------------------

def cmd_merge(args: argparse.Namespace) -> None:
    import copy
    import torch

    from llm.checkpoint import FORMAT_VERSION
    from llm.model import GPT
    from merge import (
        _check_compatible,
        _check_mergeable,
        _load_state,
        _load_tokenizer_data,
        _tokenizer_path,
        _vocab_key,
        merge_average,
        merge_slerp,
    )

    if len(args.models) < 2:
        print("[error] Need at least 2 --models to merge.")
        sys.exit(1)
    for m in args.models:
        if not os.path.exists(m):
            print(f"[error] Model not found: {m}")
            sys.exit(1)

    if args.method == "average":
        weights = args.weights if args.weights is not None else [1.0] * len(args.models)
        if len(weights) != len(args.models):
            print(f"[error] Got {len(weights)} --weights for "
                  f"{len(args.models)} --models.")
            sys.exit(1)
    else:
        weights = None

    cfgs, sds = [], []
    for m in args.models:
        cfg, sd = _load_state(m)
        try:
            _check_mergeable(sd, m)
        except ValueError as e:
            print(f"[error] {e}")
            sys.exit(1)
        cfgs.append(cfg)
        sds.append(sd)

    try:
        _check_compatible(cfgs)
    except ValueError as e:
        print(f"[error] {e}")
        sys.exit(1)

    try:
        tok_datas = [_load_tokenizer_data(m) for m in args.models]
    except FileNotFoundError as e:
        print(f"[error] {e}")
        sys.exit(1)
    ref_data = tok_datas[args.tokenizer_from]
    ref_key = _vocab_key(ref_data)
    mismatch = [i for i, d in enumerate(tok_datas)
                if _vocab_key(d) != ref_key]
    if mismatch and not args.allow_different_tokenizer:
        print(f"[error] Tokenizers differ between model {args.tokenizer_from} "
              f"and model(s) {mismatch}.")
        sys.exit(1)

    if args.method == "slerp":
        try:
            merged_sd = merge_slerp(sds, args.t)
        except ValueError as e:
            print(f"[error] {e}")
            sys.exit(1)
    else:
        assert weights is not None
        merged_sd = merge_average(sds, weights)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "config": copy.deepcopy(cfgs[0]),
        "state_dict": merged_sd,
        "merge_method": args.method,
        "merge_models": list(args.models),
    }
    torch.save(payload, args.out)
    print(f"[merge] saved → {args.out}")

    import shutil
    out_tok = CharTokenizer_default_path(args.out)
    shutil.copy2(_tokenizer_path(args.models[args.tokenizer_from]), out_tok)
    print(f"[merge] tokenizer → {out_tok}")
    GPT.load(args.out)
    print("[merge] verified: reload OK")


def CharTokenizer_default_path(model_path: str) -> str:
    from llm.tokenizer import CharTokenizer
    return CharTokenizer.default_path(model_path)


# ---------------------------------------------------------------------------
# Sub-command: gguf
# ---------------------------------------------------------------------------

def cmd_gguf(args: argparse.Namespace) -> None:
    from pth2gguf import convert

    try:
        convert(args.model, args.out, args.outtype, args.name, args.compat)
    except (FileNotFoundError, ValueError) as e:
        print(f"[error] {e}")
        sys.exit(1)
    src = os.path.getsize(args.model) / (1024 * 1024)
    dst = os.path.getsize(args.out) / (1024 * 1024)
    print(f"[gguf] {args.model} → {args.out} [{args.outtype}/{args.compat}]")
    print(f"[gguf] Size: {src:.2f} MB → {dst:.2f} MB ({dst / src * 100:.1f}%)")


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
                   help="Path to a .txt/.json/.jsonl file OR a folder of them")
    t.add_argument("--out",           default="model.pth",
                   help="Output model path (default: model.pth)")
    t.add_argument("--pattern",       default="*.txt,*.json,*.jsonl",
                   help="Comma-separated glob(s) when --data is a folder "
                        "(default: *.txt,*.json,*.jsonl)")
    t.add_argument("--text_field",    default=None,
                   help="JSON field(s) to train on, comma-separated "
                        "(default: auto-detect)")
    t.add_argument("--format",        default="auto",
                   choices=["auto", "txt", "json", "jsonl"],
                   help="Corpus parser (default: auto-detect from extension)")
    t.add_argument("--no_recurse",    action="store_true",
                   help="Do not search subfolders (top-level only)")
    t.add_argument("--tokenizer",     default="char",
                   choices=["char", "bpe"],
                   help="Tokenizer: char (default) or byte-level BPE")
    t.add_argument("--vocab-size",    type=int, default=2000,
                   help="BPE vocabulary size target (default: 2000)")
    # Model
    t.add_argument("--arch",          default="shimba",
                   choices=["shimba", "gpt2", "llama", "flash"],
                   help="Block architecture (default: shimba)")
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

    # ── merge ────────────────────────────────────────────────────────
    m = sub.add_parser("merge", help="Merge two or more checkpoints into one")
    m.add_argument("--models", nargs="+", required=True,
                   help="Two or more input .pth files")
    m.add_argument("--out", required=True, help="Output merged .pth path")
    m.add_argument("--method", choices=["average", "slerp"], default="average")
    m.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Per-model weights for average (default: equal)")
    m.add_argument("--t", type=float, default=0.5,
                   help="Interpolation factor for slerp (default: 0.5)")
    m.add_argument("--tokenizer-from", type=int, default=0)
    m.add_argument("--allow-different-tokenizer", action="store_true")

    # ── gguf ─────────────────────────────────────────────────────────
    u = sub.add_parser("gguf", help="Convert a .pth checkpoint to GGUF")
    u.add_argument("--model", required=True, help="Input model .pth file")
    u.add_argument("--out", required=True, help="Output .gguf path")
    u.add_argument("--outtype", choices=["f32", "f16", "q8_0"], default="f32")
    u.add_argument("--compat", choices=["auto", "none", "gpt2", "llama"],
                   default="auto",
                   help="GGUF architecture (default: auto follows checkpoint)")
    u.add_argument("--name", default=None,
                   help="Model name for GGUF metadata (default: input stem)")

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
        print()
        print("  # merge two checkpoints")
        print("  python setup.py merge --models a.pth b.pth --out merged.pth")
        print()
        print("  # convert to GGUF")
        print("  python setup.py gguf --model model.pth --out model.gguf --outtype q8_0")
        sys.exit(0)

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "merge":
        cmd_merge(args)
    elif args.command == "gguf":
        cmd_gguf(args)
    elif args.command == "help":
        parser.print_help()
        print("\n" + __doc__)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()