#!/usr/bin/env python3
"""
quantize.py — Reduce precision of a trained Shimba LLM model.

Usage:
    python quantize.py --model model.pth --out model_quantized.pth --dtype int8
    python quantize.py --model model.pth --out model_fp16.pth --dtype float16
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.model import GPT
from llm.compat import enable_safe_output

enable_safe_output()

def quantize_int8(model: torch.nn.Module) -> torch.nn.Module:
    from torch.quantization import quantize_dynamic
    return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

def quantize_float16(model: torch.nn.Module) -> torch.nn.Module:
    return model.half()

def main():
    parser = argparse.ArgumentParser(description="Quantize a Shimba LLM model")
    parser.add_argument("--model", required=True, help="Input model .pth file")
    parser.add_argument("--out",   required=True, help="Output quantized model path")
    parser.add_argument("--dtype", choices=["int8", "float16"], default="int8",
                        help="Quantization target (default: int8)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[error] Model not found: {args.model}")
        sys.exit(1)

    print(f"[quantize] Loading {args.model} ...")
    # GPT.load handles every historical checkpoint layout (weights under
    # "model"/"weights", _orig_mod-prefixed keys from torch.compile).
    model = GPT.load(args.model)

    print(f"[quantize] Applying {args.dtype} quantization ...")
    if args.dtype == "int8":
        quantized_model = quantize_int8(model)
    else:  # float16
        quantized_model = quantize_float16(model)

    # Save with consistent keys (using 'config' and 'state_dict' for compatibility)
    # Note: an int8-quantised module's state_dict can only be loaded back into
    # another quantised model, so the tokenizer path is carried over unchanged.
    torch.save({"config": model.cfg, "state_dict": quantized_model.state_dict()}, args.out)
    print(f"[quantize] Saved quantized model to {args.out}")

    # Show size reduction
    orig_size = os.path.getsize(args.model) / (1024 * 1024)
    new_size = os.path.getsize(args.out) / (1024 * 1024)
    print(f"[quantize] Size: {orig_size:.2f} MB → {new_size:.2f} MB ({new_size/orig_size*100:.1f}%)")

if __name__ == "__main__":
    main()