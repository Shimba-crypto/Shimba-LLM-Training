#!/usr/bin/env python3
"""
pth2gguf.py — Convert a Shimba LLM .pth checkpoint to GGUF.

Writes a GGUFv3 file with architecture "shimba" so the weights and the
char-level vocabulary travel in a single file. The tensor layout follows
the GGUF convention (dims stored reversed vs PyTorch, raw bytes unchanged).

Usage:
    python pth2gguf.py --model model.pth --out model.gguf
    python pth2gguf.py --model model.pth --out model.gguf --outtype q8_0
    python pth2gguf.py --model model.pth --out model.gguf --outtype f16

Out types:
    f32  — everything as float32 (largest, lossless)
    f16  — 2-D weights as float16, norms/biases stay float32
    q8_0 — 2-D weights block-quantized (32 vals per block), norms stay float32

Compat modes (--compat):
    none — architecture "shimba" (storage/exchange; llama.cpp cannot load it)
    gpt2 — architecture "gpt2" (loadable by llama.cpp/Ollama).
           The block layout matches GPT-2 (fused QKV, 4x GELU MLP, learned
           positions, tied output), so weights are transcribed with 2-D
           matrices transposed to GPT-2's [in, out] layout. The char-level
           vocab is kept as-is with an empty merges list, so prompts
           tokenize character by character for ASCII text. Non-ASCII bytes
           fall back to the unknown token.

Notes:
    * Requires the tokenizer JSON beside the .pth (<stem>_tokenizer.json).
    * llama.cpp does not know the "shimba" architecture, so the file is for
      storage/exchange and for runtimes that read the tensor names below —
      it is not directly loadable by stock llama.cpp.
    * Tensor names: token_embd.weight, position_embd.weight,
      blk.{i}.attn_qkv.weight, blk.{i}.attn_output.weight,
      blk.{i}.ffn_up.weight, blk.{i}.ffn_down.weight,
      blk.{i}.ln1.weight, blk.{i}.ln2.weight, output_norm.weight,
      output.weight (+ .bias variants when present).
"""

import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from llm.checkpoint import load_checkpoint
from llm.compat import enable_safe_output
from llm.tokenizer import CharTokenizer

enable_safe_output()

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

# GGUF metadata value types
VT_UINT8 = 0
VT_INT8 = 1
VT_UINT16 = 2
VT_INT16 = 3
VT_UINT32 = 4
VT_INT32 = 5
VT_FLOAT32 = 6
VT_BOOL = 7
VT_STRING = 8
VT_ARRAY = 9
VT_UINT64 = 10
VT_INT64 = 11
VT_FLOAT64 = 12

# GGML tensor types we emit
GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 8


def _pad_len(n: int, align: int = GGUF_ALIGNMENT) -> int:
    return (align - (n % align)) % align


def _write_string(buf: bytearray, s: str) -> None:
    b = s.encode("utf-8")
    buf += struct.pack("<Q", len(b))
    buf += b


def _write_val(buf: bytearray, vtype: int, value) -> None:
    if vtype == VT_UINT8:
        buf += struct.pack("<B", value)
    elif vtype == VT_INT8:
        buf += struct.pack("<b", value)
    elif vtype == VT_UINT16:
        buf += struct.pack("<H", value)
    elif vtype == VT_INT16:
        buf += struct.pack("<h", value)
    elif vtype == VT_UINT32:
        buf += struct.pack("<I", value)
    elif vtype == VT_INT32:
        buf += struct.pack("<i", value)
    elif vtype == VT_FLOAT32:
        buf += struct.pack("<f", value)
    elif vtype == VT_BOOL:
        buf += struct.pack("<B", 1 if value else 0)
    elif vtype == VT_STRING:
        _write_string(buf, value)
    elif vtype == VT_UINT64:
        buf += struct.pack("<Q", value)
    elif vtype == VT_INT64:
        buf += struct.pack("<q", value)
    elif vtype == VT_FLOAT64:
        buf += struct.pack("<d", value)
    else:
        raise ValueError(f"unsupported value type {vtype}")


def _write_kv(buf: bytearray, key: str, vtype: int, value) -> None:
    _write_string(buf, key)
    buf += struct.pack("<I", vtype)
    if vtype == VT_ARRAY:
        elem_type, items = value
        buf += struct.pack("<I", elem_type)
        buf += struct.pack("<Q", len(items))
        for it in items:
            if elem_type == VT_STRING:
                _write_string(buf, it)
            else:
                _write_val(buf, elem_type, it)
    else:
        _write_val(buf, vtype, value)


def _storage_bytes(t: torch.Tensor) -> bytes:
    t = t.contiguous().cpu()
    try:
        return t.untyped_storage().tobytes()
    except Exception:
        return t.numpy().tobytes()


def _to_f32_bytes(t: torch.Tensor) -> bytes:
    return _storage_bytes(t.float())


def _to_f16_bytes(t: torch.Tensor) -> bytes:
    return _storage_bytes(t.half())


# Block quantization lives in llm.scw (shared with the .scw format).
from llm.scw import quantize_q8_0 as _quantize_q8_0


def map_names(state_dict: dict, cfg, compat: str = "none"):
    """Map checkpoint keys to GGUF tensor names. Skips buffers."""
    n_head = cfg.n_head
    n_kv = getattr(cfg, "n_head_kv", None) or n_head
    head_dim = cfg.n_embd // n_head
    mapped = []
    for k, v in state_dict.items():
        if "causal_mask" in k:
            continue
        if not isinstance(v, torch.Tensor) or not v.is_floating_point():
            continue
        if k == "transformer.wte.weight":
            mapped.append(("token_embd.weight", v))
            continue
        if k == "transformer.wpe.weight":
            # Learned positions exist only for shimba/gpt2 archs; llama.cpp
            # has no position_embd tensor for the llama arch (RoPE instead).
            if compat != "llama":
                mapped.append(("position_embd.weight", v))
            continue
        if k == "transformer.ln_f.weight":
            mapped.append(("output_norm.weight", v))
            continue
        if k == "transformer.ln_f.bias":
            mapped.append(("output_norm.bias", v))
            continue
        if k in ("lm_head.weight",):
            mapped.append(("output.weight", v))
            continue
        if k == "lm_head.bias":
            mapped.append(("output.bias", v))
            continue
        if not k.startswith("transformer.h."):
            mapped.append(("shimba." + k.replace(".", "_"), v))
            continue
        # transformer.h.{i}.{rest}
        parts = k.split(".")
        try:
            idx = int(parts[2])
        except (ValueError, IndexError):
            mapped.append(("shimba." + k.replace(".", "_"), v))
            continue
        rest = ".".join(parts[3:])
        if compat == "llama" and rest == "attn.c_attn.weight":
            # llama.cpp wants split q/k/v (no fused tensor for this arch).
            q_len, kv_len = n_head * head_dim, n_kv * head_dim
            q, kq, vq = v.split((q_len, kv_len, kv_len), dim=0)
            mapped.append((f"blk.{idx}.attn_q.weight", q))
            mapped.append((f"blk.{idx}.attn_k.weight", kq))
            mapped.append((f"blk.{idx}.attn_v.weight", vq))
            continue
        if compat in ("gpt2", "llama"):
            table = {
                "ln1.weight": f"blk.{idx}.attn_norm.weight",
                "ln1.bias": f"blk.{idx}.attn_norm.bias",
                "attn.c_attn.weight": f"blk.{idx}.attn_qkv.weight",
                "attn.c_attn.bias": f"blk.{idx}.attn_qkv.bias",
                "attn.c_proj.weight": f"blk.{idx}.attn_output.weight",
                "attn.c_proj.bias": f"blk.{idx}.attn_output.bias",
                "ln2.weight": f"blk.{idx}.ffn_norm.weight",
                "ln2.bias": f"blk.{idx}.ffn_norm.bias",
                "mlp.fc.weight": f"blk.{idx}.ffn_up.weight",
                "mlp.fc.bias": f"blk.{idx}.ffn_up.bias",
                "mlp.gate.weight": f"blk.{idx}.ffn_gate.weight",
                "mlp.gate.bias": f"blk.{idx}.ffn_gate.bias",
                "mlp.up.weight": f"blk.{idx}.ffn_up.weight",
                "mlp.up.bias": f"blk.{idx}.ffn_up.bias",
                "mlp.proj.weight": f"blk.{idx}.ffn_down.weight",
                "mlp.proj.bias": f"blk.{idx}.ffn_down.bias",
            }
        else:
            table = {
                "ln1.weight": f"blk.{idx}.ln1.weight",
                "ln1.bias": f"blk.{idx}.ln1.bias",
                "attn.c_attn.weight": f"blk.{idx}.attn_qkv.weight",
                "attn.c_attn.bias": f"blk.{idx}.attn_qkv.bias",
                "attn.c_proj.weight": f"blk.{idx}.attn_output.weight",
                "attn.c_proj.bias": f"blk.{idx}.attn_output.bias",
                "ln2.weight": f"blk.{idx}.ln2.weight",
                "ln2.bias": f"blk.{idx}.ln2.bias",
                "mlp.fc.weight": f"blk.{idx}.ffn_up.weight",
                "mlp.fc.bias": f"blk.{idx}.ffn_up.bias",
                "mlp.gate.weight": f"blk.{idx}.ffn_gate.weight",
                "mlp.gate.bias": f"blk.{idx}.ffn_gate.bias",
                "mlp.up.weight": f"blk.{idx}.ffn_up.weight",
                "mlp.up.bias": f"blk.{idx}.ffn_up.bias",
                "mlp.proj.weight": f"blk.{idx}.ffn_down.weight",
                "mlp.proj.bias": f"blk.{idx}.ffn_down.bias",
            }
        name = table.get(rest, "shimba." + k.replace(".", "_"))
        # No transposes anywhere: llama.cpp reads dense matrices in the same
        # [out, in] arrangement our PyTorch Linears already use (verified
        # against its shape checks for gpt2; llama matches HF Linear layout).
        mapped.append((name, v))
    if compat == "gpt2":
        # GPT-2 LayerNorms and dense layers carry biases; ours are
        # bias-free, so emit matching zero biases (mathematically identical,
        # keeps the file loadable).
        by_name = dict(mapped)
        extra = []
        for name, w in mapped:
            if not name.endswith(".weight"):
                continue
            base = name[:-len(".weight")]
            leaf = base.split(".")[-1]
            if not (leaf in ("attn_qkv", "attn_output", "attn_norm",
                             "ffn_norm", "ffn_up", "ffn_down")
                    or base == "output_norm"):
                continue
            dim = w.shape[0]
            extra.append((base + ".bias", torch.zeros(dim)))
        mapped.extend(extra)
    # Deterministic order so repeated conversions are byte-identical.
    mapped.sort(key=lambda x: x[0])
    return mapped


def _gpt2_byte_encode(s: str) -> str:
    """
    Transcribe a single ASCII character to its GPT-2 byte-encoding form.

    llama.cpp byte-encodes prompt bytes before BPE matching, so a raw " "
    would never match a vocab entry. Mapping " " -> "Ġ" (U+0120) keeps the
    same token id while making the byte-encoder land on it. Multi-byte
    characters are left alone (unreachable via byte fallback, same as now).
    """
    b = s.encode("utf-8")
    if len(b) != 1:
        return s
    v = b[0]
    if 33 <= v <= 126 or 161 <= v <= 255:
        return chr(v)
    # Mirrors HF's bytes_to_unicode: bytes outside the printable ranges
    # map to U+0100 and up, in byte order.
    n = 0
    for i in range(256):
        if not (33 <= i <= 126 or 161 <= i <= 255):
            if i == v:
                return chr(256 + n)
            n += 1
    return s  # unreachable


def build_metadata(cfg, tokenizer: CharTokenizer, model_name: str,
                     compat: str = "none"):
    arch = compat if compat in ("gpt2", "llama") else "shimba"
    n_kv = getattr(cfg, "n_head_kv", None) or cfg.n_head
    head_dim = cfg.n_embd // cfg.n_head
    kvs = []
    kvs.append(("general.architecture", VT_STRING, arch))
    kvs.append(("general.name", VT_STRING, model_name))
    kvs.append(("general.quantization_version", VT_UINT32, 2))
    kvs.append((f"{arch}.block_count", VT_UINT32, int(cfg.n_layer)))
    kvs.append((f"{arch}.embedding_length", VT_UINT32, int(cfg.n_embd)))
    kvs.append((f"{arch}.context_length", VT_UINT32, int(cfg.block_size)))
    kvs.append((f"{arch}.attention.head_count", VT_UINT32, int(cfg.n_head)))
    kvs.append((f"{arch}.attention.head_count_kv", VT_UINT32, int(n_kv)))
    kvs.append((f"{arch}.vocab_size", VT_UINT32, int(cfg.vocab_size)))
    kvs.append((f"{arch}.feed_forward_length", VT_UINT32, int(4 * cfg.n_embd)))
    try:
        dropout = float(cfg.dropout)
    except Exception:
        dropout = None
    if compat == "none" and dropout is not None:
        kvs.append(("shimba.dropout", VT_FLOAT32, dropout))
    if compat == "none":
        kvs.append(("shimba.tie_word_embeddings", VT_BOOL, True))
    if compat == "gpt2":
        kvs.append(("gpt2.attention.layer_norm_epsilon", VT_FLOAT32, 1e-5))
    if compat == "llama":
        kvs.append(("llama.attention.layer_norm_rms_epsilon", VT_FLOAT32, 1e-5))
        kvs.append(("llama.rope.dimension_count", VT_UINT32, int(head_dim)))
        rope_base = getattr(cfg, "rope_base", 10000.0) or 10000.0
        kvs.append(("llama.rope.freq_base", VT_FLOAT32, float(rope_base)))

    # Char-level vocab, ordered by id.
    tokens = [tokenizer.idx2char[i] for i in range(tokenizer.vocab_size)]
    # Both llama.cpp-loadable modes share the BPE/char trick: an empty merges
    # list keeps every word split into single characters, and the GPT-2 byte
    # transcription makes the byte-encoder land on those ids. (A native
    # SPM/llama tokenizer would throw on any byte missing from the vocab.)
    # A trained BPE tokenizer already stores byte-encoded strings plus real
    # merges, so it passes through untouched.
    bpe_chars = compat in ("gpt2", "llama")
    real_merges = getattr(tokenizer, "merges", None) or None
    if bpe_chars and not real_merges:
        tokens = [_gpt2_byte_encode(t) for t in tokens]
    types = []
    for i in range(tokenizer.vocab_size):
        if i == 0:
            types.append(3)  # control (PAD)
        elif i == 1:
            types.append(2)  # unknown (UNK)
        else:
            types.append(1)  # normal
    if compat == "llama":
        # Weights use the llama layout; the tokenizer reuses the working
        # BPE/char setup (see above).
        kvs.append(("tokenizer.ggml.model", VT_STRING, "gpt2"))
        kvs.append(("tokenizer.ggml.pre", VT_STRING, "gpt-2"))
    elif compat == "gpt2":
        kvs.append(("tokenizer.ggml.model", VT_STRING, "gpt2"))
        kvs.append(("tokenizer.ggml.pre", VT_STRING, "gpt-2"))
    else:
        kvs.append(("tokenizer.ggml.model", VT_STRING, "shimba-char"))
        kvs.append(("tokenizer.ggml.pre", VT_STRING, "shimba-char"))
    kvs.append(("tokenizer.ggml.tokens", VT_ARRAY, (VT_STRING, tokens)))
    kvs.append(("tokenizer.ggml.token_type", VT_ARRAY, (VT_INT32, types)))
    if bpe_chars:
        if real_merges:
            kvs.append(("tokenizer.ggml.merges", VT_ARRAY,
                        (VT_STRING, [" ".join(m) for m in real_merges])))
        else:
            # No BPE merges: with an empty list every word stays split into
            # single characters, which then match the char-level vocab below.
            kvs.append(("tokenizer.ggml.merges", VT_ARRAY, (VT_STRING, [])))
    kvs.append(("tokenizer.ggml.unknown_token_id", VT_UINT32, 1))
    kvs.append(("tokenizer.ggml.padding_token_id", VT_UINT32, 0))
    kvs.append(("tokenizer.ggml.add_bos_token", VT_BOOL, False))
    kvs.append(("tokenizer.ggml.add_eos_token", VT_BOOL, False))
    return kvs


def convert(model_path: str, out_path: str, outtype: str = "f32",
            name: str | None = None, compat: str = "none") -> str:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    tok_path = CharTokenizer.default_path(model_path)
    if not os.path.exists(tok_path):
        raise FileNotFoundError(
            f"Tokenizer not found at '{tok_path}'. "
            f"The tokenizer JSON must sit beside the model file."
        )
    outtype = outtype.lower()
    if outtype not in ("f32", "f16", "q8_0"):
        raise ValueError(f"--outtype must be f32, f16 or q8_0 (got {outtype})")
    compat = (compat or "auto").lower()
    if compat not in ("auto", "none", "gpt2", "llama"):
        raise ValueError(f"--compat must be auto, none, gpt2 or llama "
                         f"(got {compat})")

    cfg, ckpt = load_checkpoint(model_path, map_location="cpu")
    sd = ckpt["state_dict"]
    # Quantized exports have no meaningful fp32 weights to convert.
    for k, v in sd.items():
        if "causal_mask" in k:
            continue
        if isinstance(v, torch.Tensor) and not v.is_floating_point():
            raise ValueError(
                f"Cannot convert '{k}' with dtype {v.dtype} — "
                f"convert the fp32 original, not a quantized export."
            )

    from llm.bpe import load_tokenizer
    tokenizer = load_tokenizer(tok_path)
    if tokenizer.vocab_size != cfg.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size={tokenizer.vocab_size} does not match "
            f"model vocab_size={cfg.vocab_size}."
        )

    model_name = name or os.path.splitext(os.path.basename(model_path))[0]
    if compat == "auto":
        # Follow the checkpoint: gpt2/llama checkpoints already match the
        # GGUF layout of that name; shimba stays custom.
        arch = getattr(cfg, "arch", "shimba") or "shimba"
        compat = arch if arch in ("gpt2", "llama") else "none"
    if compat == "llama" and (getattr(cfg, "arch", "shimba") or "shimba") != "llama":
        print("[warn] --compat llama with a non-llama checkpoint; "
              "the blocks differ (RoPE/SwiGLU), output will not match.")
    mapped = map_names(sd, cfg, compat)
    if not mapped:
        raise ValueError("No convertible tensors found in checkpoint.")

    # --- encode tensor data -------------------------------------------
    enc = []  # (name, dims_reversed, ggml_type, bytes)
    for tname, t in mapped:
        is_1d = t.dim() <= 1
        if outtype == "f32":
            gtype, blob = GGML_F32, _to_f32_bytes(t)
        elif outtype == "f16":
            if is_1d:
                gtype, blob = GGML_F32, _to_f32_bytes(t)
            else:
                gtype, blob = GGML_F16, _to_f16_bytes(t)
        else:  # q8_0
            if is_1d:
                gtype, blob = GGML_F32, _to_f32_bytes(t)
            else:
                gtype, blob = GGML_Q8_0, _quantize_q8_0(t)
        dims_rev = list(reversed(list(t.shape)))
        enc.append((tname, dims_rev, gtype, blob))

    kvs = build_metadata(cfg, tokenizer, model_name, compat)

    # --- layout: offsets relative to tensor-data start ------------------
    offset = 0
    offsets = []
    for _, _, _, blob in enc:
        offsets.append(offset)
        offset += len(blob) + _pad_len(len(blob))

    buf = bytearray()
    buf += GGUF_MAGIC
    buf += struct.pack("<I", GGUF_VERSION)
    buf += struct.pack("<Q", len(enc))
    buf += struct.pack("<Q", len(kvs))
    for key, vtype, value in kvs:
        _write_kv(buf, key, vtype, value)
    for (tname, dims_rev, gtype, _), off in zip(enc, offsets):
        _write_string(buf, tname)
        buf += struct.pack("<I", len(dims_rev))
        for d in dims_rev:
            buf += struct.pack("<Q", int(d))
        buf += struct.pack("<I", gtype)
        buf += struct.pack("<Q", off)
    buf += b"\x00" * _pad_len(len(buf))
    for _, _, _, blob in enc:
        buf += blob
        buf += b"\x00" * _pad_len(len(blob))

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(buf)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Convert Shimba .pth to GGUF")
    p.add_argument("--model", required=True, help="Input model .pth file")
    p.add_argument("--out", required=True, help="Output .gguf path")
    p.add_argument("--outtype", choices=["f32", "f16", "q8_0"], default="f32",
                   help="Weight precision in the GGUF file (default: f32)")
    p.add_argument("--name", default=None,
                   help="Model name for GGUF metadata (default: input stem)")
    p.add_argument("--compat", choices=["auto", "none", "gpt2", "llama"],
                   default="auto",
                   help="GGUF architecture: auto follows the checkpoint arch, "
                        "none writes custom 'shimba', gpt2/llama force one "
                        "(loadable by llama.cpp/Ollama; default: auto)")
    args = p.parse_args()

    try:
        convert(args.model, args.out, args.outtype, args.name, args.compat)
    except (FileNotFoundError, ValueError) as e:
        print(f"[error] {e}")
        sys.exit(1)

    src = os.path.getsize(args.model) / (1024 * 1024)
    dst = os.path.getsize(args.out) / (1024 * 1024)
    print(f"[gguf] {args.model} → {args.out} [{args.outtype}/{args.compat}]")
    print(f"[gguf] Size: {src:.2f} MB → {dst:.2f} MB ({dst / src * 100:.1f}%)")


if __name__ == "__main__":
    main()
