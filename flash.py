#!/usr/bin/env python3
"""
flash.py — give a trained model drugs so it runs fast.

    python flash.py pack --model model.pth --out model_flash.pth
    python flash.py pack --model model.pth --out model_flash.pth --dtype int8
    python flash.py info --model model_flash.pth
    python flash.py generate --model model_flash.pth --prompt "Once upon a time"

pack:      copies the checkpoint to *_flash.pth with inference drugs
           (KV-cache metadata always; fp16 or int8 weights on request)
           and carries the tokenizer JSON alongside the output.
info:      shows arch, params, drugs, and cache cost per token.
generate:  decodes through the incremental KV-cache path (fast_generate).
           Any arch works; `--arch flash` checkpoints cache the least.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.compat import enable_safe_output

enable_safe_output()


def cmd_pack(args):
    from llm.flash import pack_flash

    if not os.path.exists(args.model):
        print(f"[error] Model not found: {args.model}")
        sys.exit(1)
    print(f"[flash] packing {args.model} → {args.out} (dtype={args.dtype}) ...")
    meta = pack_flash(args.model, args.out, dtype=args.dtype)
    orig_mb = os.path.getsize(args.model) / 1024 ** 2
    new_mb = os.path.getsize(args.out) / 1024 ** 2
    print(f"[flash] drugs: {'+'.join(meta['drugs'])}")
    print(f"[flash] size: {orig_mb:.1f} MB → {new_mb:.1f} MB "
          f"({new_mb / orig_mb * 100:.0f}%)")
    print(f"[flash] wrote {args.out} (+ tokenizer)")


def cmd_info(args):
    from llm.flash import flash_info

    if not os.path.exists(args.model):
        print(f"[error] Model not found: {args.model}")
        sys.exit(1)
    print(flash_info(args.model))


def cmd_generate(args):
    import torch
    from llm.bpe import load_tokenizer
    from llm.device import resolve_device
    from llm.flash import fast_generate, load_flash
    from llm.tokenizer import CharTokenizer

    if not os.path.exists(args.model):
        print(f"[error] Model not found: {args.model}")
        sys.exit(1)
    device = resolve_device(args.device)
    model, meta = load_flash(args.model, device=device)
    if meta:
        print(f"[flash] drugs: {'+'.join(meta.get('drugs', []))}")
    if args.compile:
        try:
            model = torch.compile(model, dynamic=True)
            print("[flash] torch.compile on (first tokens slow, then fast)")
        except Exception as e:
            print(f"[flash] compile skipped ({e})")

    tok_path = CharTokenizer.default_path(args.model)
    tok = load_tokenizer(tok_path)

    stops = args.stop or []
    out = fast_generate(
        model, tok, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop_strings=stops,
    )
    print(out)


def main():
    p = argparse.ArgumentParser(description="Shimba LLM — flash inference kit")
    sub = p.add_subparsers(dest="cmd", required=True)

    pk = sub.add_parser("pack", help="Pack a *_flash.pth from a checkpoint")
    pk.add_argument("--model", required=True)
    pk.add_argument("--out", required=True)
    pk.add_argument("--dtype", choices=["f32", "f16", "int8"], default="f16")
    pk.set_defaults(func=cmd_pack)

    inf = sub.add_parser("info", help="Show model + flash status")
    inf.add_argument("--model", required=True)
    inf.set_defaults(func=cmd_info)

    gen = sub.add_parser("generate", help="Generate via the KV-cache fast path")
    gen.add_argument("--model", required=True)
    gen.add_argument("--prompt", default="Once upon a time")
    gen.add_argument("--max_tokens", type=int, default=200)
    gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--top_k", type=int, default=40)
    gen.add_argument("--top_p", type=float, default=0.95)
    gen.add_argument("--repetition_penalty", type=float, default=1.1)
    gen.add_argument("--stop", action="append", default=None,
                     help="Stop string (repeatable)")
    gen.add_argument("--device", default="auto")
    gen.add_argument("--compile", action="store_true",
                     help="torch.compile the model first")
    gen.set_defaults(func=cmd_generate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
