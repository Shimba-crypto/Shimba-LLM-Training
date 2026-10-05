#!/usr/bin/env python3
"""
flash_slms.py — the flash small-model family: pack + verify 10M/50M/100M.

    python flash_slms.py specs
    python flash_slms.py pack --model m.pth --out m_flash.pth [--dtype f16]
    python flash_slms.py verify --model m_flash.pth [--probes 60]

specs:  print the three family configs (flash arch, exact param counts).
pack:   *_flash.pth via llm.flash (KV-cache metadata, fp16/int8 optional).
verify: generate probe answers through the KV-cache fast path and score
        exact-match, with vida re-deriving every expected value as an
        independent second opinion (falls back to Python when vendor/vida
        is not built).

The family (flash arch, block 256, char-scale vocab 256):

    flash-10M   256x12x8    ~11.5M params   laptop / CPU inference
    flash-50M   512x14x8    ~53.4M params   single T4, the demo size
    flash-100M  768x12x12  ~102.8M params   needs real training data
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.compat import enable_safe_output

enable_safe_output()

SPECS = {
    "flash-10M": dict(n_embd=256, n_layer=12, n_head=8, block_size=256,
                      use="laptop / CPU inference"),
    "flash-50M": dict(n_embd=512, n_layer=14, n_head=8, block_size=256,
                      use="single T4, the demo size"),
    "flash-100M": dict(n_embd=768, n_layer=12, n_head=12, block_size=256,
                       use="needs real training data"),
}


def cmd_specs(_args):
    from llm.flash import cache_bytes_per_token, describe_flash
    from llm.model import GPT, GPTConfig
    print(f"  {'name':<10s} {'config':<12s} {'params':>10s}  cache/tok  use")
    for name, spec in SPECS.items():
        cfg = GPTConfig(vocab_size=256, arch="flash", **{k: v for k, v in
                                                         spec.items() if k != "use"})
        n = sum(p.numel() for p in GPT(cfg).parameters())
        kb = cache_bytes_per_token(cfg) / 1024
        print(f"  {name:<10s} {spec['n_embd']}x{spec['n_layer']}x{spec['n_head']:<6d} "
              f"{n:>10,}  {kb:>6.1f} KB  {spec['use']}")
        _ = describe_flash(cfg)


def cmd_pack(args):
    from llm.flash import pack_flash
    if not os.path.exists(args.model):
        print(f"[error] Model not found: {args.model}")
        sys.exit(1)
    meta = pack_flash(args.model, args.out, dtype=args.dtype)
    size = os.path.getsize(args.out) / 1024 ** 2
    print(f"[flash-slms] drugs={'+'.join(meta['drugs'])} "
          f"{size:.1f} MB -> {args.out} (+ tokenizer)")


def build_probes(n: int, seed: int = 7):
    """(prompt, vida_expr, expected) triples with Python-computed answers."""
    rng = random.Random(seed)
    probes = []
    for _ in range(n):
        kind = rng.randrange(4)
        if kind == 0:
            a, b = rng.randint(2, 99), rng.randint(2, 99)
            probes.append((f"Instruction: What is {a} x {b}?\nResponse:",
                           f"{a}*{b}", str(a * b)))
        elif kind == 1:
            a = rng.randint(2, 50)
            probes.append((f"Instruction: What is {a} squared?\nResponse:",
                           f"{a}*{a}", str(a * a)))
        elif kind == 2:
            b, q = rng.randint(2, 25), rng.randint(2, 60)
            probes.append((f"Instruction: What is {b * q} / {b}?\nResponse:",
                           f"{b * q}/{b}", str(q)))
        else:
            # No int-division: vda truncates ints, Python does not.
            a = rng.randint(11, 999)
            b = rng.randint(11, 999)
            probes.append((f"Instruction: What is {a} + {b}?\nResponse:",
                           f"{a}+{b}", str(a + b)))
    return probes


def cmd_verify(args):
    from llm.bpe import load_tokenizer
    from llm.device import resolve_device
    from llm.flash import fast_generate, load_flash
    from llm.tokenizer import CharTokenizer
    from llm.vida_compute import vda_available, verify_rows

    if not os.path.exists(args.model):
        print(f"[error] Model not found: {args.model}")
        sys.exit(1)
    device = resolve_device(args.device)
    model, meta = load_flash(args.model, device=device)
    if meta:
        print(f"[flash-slms] drugs: {'+'.join(meta.get('drugs', []))}")
    print(f"[flash-slms] vida engine: "
          f"{'vda binary' if vda_available() else 'python fallback'}")

    tok = load_tokenizer(CharTokenizer.default_path(args.model))
    probes = build_probes(args.probes)

    # Second opinion first: vida re-derives every expected answer. A
    # mismatch here is a probe bug, never a model miss.
    second = verify_rows([(e, x) for _, e, x in probes])
    print(f"[flash-slms] probe integrity: {second['hits']}/{second['total']} "
          f"via {second['engine']}")
    for expr, expected, got in second["mismatches"]:
        print(f"  PROBE-BAD {expr} expected={expected} engine={got}")

    hits, total, shown = 0, 0, 0
    for prompt, _expr, expected in probes:
        out = fast_generate(model, tok, prompt,
                            max_new_tokens=args.max_tokens, temperature=0.0,
                            repetition_penalty=1.0, top_k=0, top_p=1.0)
        tail = out[len(prompt):].strip().splitlines()
        ans = next((ln for ln in reversed(tail) if ln.startswith("Answer:")), "")
        ok = expected in ans
        hits += ok
        total += 1
        if (not ok and shown < 5) or args.verbose:
            shown += 1
            print(f"  {'HIT ' if ok else 'MISS'} want={expected} got={ans!r}")
    print(f"\n[flash-slms] exact-match: {hits}/{total}")


def main():
    p = argparse.ArgumentParser(description="flash-slms: pack + verify small models")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("specs", help="Print the family configs")
    s.set_defaults(func=cmd_specs)

    k = sub.add_parser("pack", help="Pack a *_flash.pth from a checkpoint")
    k.add_argument("--model", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--dtype", choices=["f32", "f16", "int8"], default="f16")
    k.set_defaults(func=cmd_pack)

    v = sub.add_parser("verify", help="Probe a model, vida double-checks answers")
    v.add_argument("--model", required=True)
    v.add_argument("--probes", type=int, default=60)
    v.add_argument("--max_tokens", type=int, default=60)
    v.add_argument("--device", default="auto")
    v.add_argument("--verbose", action="store_true")
    v.set_defaults(func=cmd_verify)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
