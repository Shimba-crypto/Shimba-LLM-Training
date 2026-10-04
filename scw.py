#!/usr/bin/env python3
"""
scw.py — Pack, inspect and unpack Shimba Container Weights (.scw).

A .scw file ships the FULL model (config + tokenizer + weights) in one
file with no pickle, loading memory-mapped for fast startup:

    python scw.py pack   --model model.pth --out model.scw
    python scw.py pack   --model model.pth --out model_q8.scw --dtype q8_0
    python scw.py info   --model model.scw
    python scw.py unpack --model model.scw --out model.pth

The .scw loads anywhere a .pth does (chat.py, setup.py generate, ...).
See llm/scw.py for the byte-level format contract.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.compat import enable_safe_output

enable_safe_output()


def cmd_pack(args: argparse.Namespace) -> None:
    from llm.scw import pack_model

    try:
        out = pack_model(args.model, args.out, weight_dtype=args.dtype,
                         name=args.name)
    except (FileNotFoundError, ValueError) as e:
        print(f"[error] {e}")
        sys.exit(1)
    src = os.path.getsize(args.model) / (1024 * 1024)
    dst = os.path.getsize(out) / (1024 * 1024)
    print(f"[scw] packed {args.model} → {out} [{args.dtype}]")
    print(f"[scw] Size: {src:.2f} MB → {dst:.2f} MB ({dst / src * 100:.1f}%)")


def cmd_info(args: argparse.Namespace) -> None:
    from llm.scw import scw_info

    try:
        info = scw_info(args.model)
    except (FileNotFoundError, ValueError) as e:
        print(f"[error] {e}")
        sys.exit(1)
    print(f"  name:      {info['name']}")
    print(f"  format:    scw v{info['version']}  weights: {info['weight_dtype']}")
    print(f"  arch:      {info['arch']}")
    print(f"  tokenizer: {info['tokenizer']}  vocab_size={info['vocab_size']}")
    print(f"  params:    {info['params']:,}")
    print(f"  tensors:   {info['tensors']}  {info['bytes_by_dtype']}")
    print(f"  file:      {info['file_mb']:.2f} MB")


def cmd_unpack(args: argparse.Namespace) -> None:
    from llm.scw import unpack_model

    try:
        pth, tok = unpack_model(args.model, args.out)
    except (FileNotFoundError, ValueError) as e:
        print(f"[error] {e}")
        sys.exit(1)
    print(f"[scw] unpacked → {pth}")
    print(f"[scw] tokenizer → {tok}")


def main() -> None:
    p = argparse.ArgumentParser(description="Shimba Container Weights (.scw)")
    sub = p.add_subparsers(dest="command", metavar="COMMAND", required=True)

    pk = sub.add_parser("pack", help="Pack a .pth (+ tokenizer) into one .scw")
    pk.add_argument("--model", required=True, help="Input model .pth file")
    pk.add_argument("--out", required=True, help="Output .scw path")
    pk.add_argument("--dtype", choices=["f32", "f16", "q8_0"], default="f32",
                    help="Weight storage (default: f32)")
    pk.add_argument("--name", default=None,
                    help="Model name in the manifest (default: input stem)")

    inf = sub.add_parser("info", help="Show a .scw manifest summary")
    inf.add_argument("--model", required=True, help="Input .scw file")

    un = sub.add_parser("unpack", help="Unpack a .scw back to .pth + tokenizer")
    un.add_argument("--model", required=True, help="Input .scw file")
    un.add_argument("--out", required=True, help="Output .pth path")

    args = p.parse_args()
    if args.command == "pack":
        cmd_pack(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "unpack":
        cmd_unpack(args)


if __name__ == "__main__":
    main()
