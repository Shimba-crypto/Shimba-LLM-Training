"""
scw.py — Shimba Container Weights (.scw), the single-file model format.

A .scw file ships the FULL model: config, tokenizer (char or BPE) and
every weight tensor in one file, with no pickle anywhere. Layout
(little-endian)::

    magic          4 bytes  "SCW1"
    version        u32      (=1)
    manifest_len   u64
    manifest       JSON, manifest_len bytes
    <pad to 64 B>  zeros
    tensor data    raw blobs back to back, each padded to 64 B

Manifest::

    {"format": "scw", "version": 1,
     "weight_dtype": "f32" | "f16" | "q8_0",
     "config": {GPTConfig fields},
     "tokenizer": {tokenizer JSON content},
     "tensors": [{"name":..., "dtype": "f32"|"f16"|"q8_0",
                  "shape": [...], "numel": N,
                  "offset": <bytes from data start>, "nbytes": M}],
     "data_size": <tensor region bytes>,
     "sha256": <hex of tensor region>}

Why it chats faster than .pth:

    * One sequential read (or mmap) instead of unpickling.
    * f32 tensors load as zero-copy views, no duplicate RAM.
    * No second file to find: the tokenizer rides inside.

The reference CLI is `scw.py` (pack / info / unpack). `slm-chat` and any
other tool can rely on `pack_model`, `load_scw`, `load_tokenizer_for`
and `unpack_model` below; the layout above is the stability contract.
"""

import hashlib
import json
import mmap
import os
import struct

import torch

MAGIC = b"SCW1"
VERSION = 1
ALIGN = 64

Q8_BLOCK = 32
Q8_BLOCK_BYTES = 2 + Q8_BLOCK  # fp16 scale + 32x int8

_STORE_DTYPES = ("f32", "f16", "q8_0")


def _pad_len(n: int, align: int = ALIGN) -> int:
    return (align - (n % align)) % align


# ---------------------------------------------------------------------------
# Q8_0 block quantization (shared with pth2gguf)
# ---------------------------------------------------------------------------

def quantize_q8_0(t: torch.Tensor) -> bytes:
    """float32 tensor -> Q8_0 blocks (fp16 scale + 32 int8 each)."""
    flat = t.float().contiguous().cpu().flatten()
    n = flat.numel()
    n_blocks = (n + Q8_BLOCK - 1) // Q8_BLOCK
    padded_len = n_blocks * Q8_BLOCK
    if padded_len != n:
        flat = torch.cat([flat, torch.zeros(padded_len - n, dtype=torch.float32)])
    blocks = flat.reshape(n_blocks, Q8_BLOCK)
    amax = blocks.abs().max(dim=1).values
    scale = amax / 127.0
    safe = scale.clone()
    safe[safe == 0] = 1.0
    q = torch.round(blocks / safe.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    q[amax == 0] = 0
    scales_b = _tobytes(scale.half())
    qs_b = _tobytes(q)
    out = bytearray(n_blocks * Q8_BLOCK_BYTES)
    for i in range(n_blocks):
        o = i * Q8_BLOCK_BYTES
        out[o:o + 2] = scales_b[i * 2:i * 2 + 2]
        out[o + 2:o + Q8_BLOCK_BYTES] = qs_b[i * Q8_BLOCK:i * Q8_BLOCK + Q8_BLOCK]
    return bytes(out)


def dequantize_q8_0(blob: bytes, numel: int, chunk_blocks: int = 4096) -> torch.Tensor:
    """
    Q8_0 blocks -> float32 tensor (truncated to numel).

    Decoded in small chunks so peak extra RAM stays flat (~1 MB) no matter
    how big the tensor is. `d` must stay 1-D: unsqueezing a (B,1) scale to
    (B,1,1) broadcasts q (B,32) into (B,B,32) — that bug once tried to
    allocate 2.7 GB for a 0.6 MB tensor. See quick_test for the lock-in.
    """
    raw = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
    n_blocks = raw.numel() // Q8_BLOCK_BYTES
    out = torch.empty(numel, dtype=torch.float32)
    pos = 0
    for s in range(0, n_blocks, chunk_blocks):
        e = min(s + chunk_blocks, n_blocks)
        blk = raw[s * Q8_BLOCK_BYTES:e * Q8_BLOCK_BYTES].reshape(-1, Q8_BLOCK_BYTES)
        d = blk[:, :2].contiguous().view(torch.float16).float().reshape(-1)
        q = blk[:, 2:].contiguous().view(torch.int8).float()
        vals = (q * d.unsqueeze(1)).reshape(-1)
        n = min(vals.numel(), numel - pos)
        out[pos:pos + n] = vals[:n]
        pos += n
        if pos >= numel:
            break
    return out


# ---------------------------------------------------------------------------
# Pack (.pth -> .scw)
# ---------------------------------------------------------------------------

def _tobytes(t: torch.Tensor) -> bytes:
    """Raw little-endian bytes of a contiguous CPU tensor, no numpy needed."""
    t = t.contiguous().cpu()
    try:
        return t.untyped_storage().tobytes()
    except AttributeError:
        pass
    try:
        return t.numpy().tobytes()
    except Exception:
        return bytes(t.view(torch.uint8).flatten().tolist())


def _to_blob(t: torch.Tensor, dtype: str) -> bytes:
    t = t.contiguous().cpu()
    if dtype == "f32":
        return _tobytes(t.float())
    if dtype == "f16":
        return _tobytes(t.half())
    if dtype == "q8_0":
        return quantize_q8_0(t)
    raise ValueError(f"unknown weight dtype {dtype!r}")


def pack_model(model_path: str, out_path: str, weight_dtype: str = "f32",
               name: str | None = None) -> str:
    """
    Pack a .pth checkpoint (+ sidecar tokenizer JSON) into one .scw file.

    weight_dtype selects tensor storage: f32 (lossless), f16 (halved size,
    1-D norms stay f32) or q8_0 (block-quantized 2-D weights, norms f32).
    """
    from .bpe import load_tokenizer as _load_tok
    from .checkpoint import load_checkpoint
    from .tokenizer import CharTokenizer
    import dataclasses

    if weight_dtype not in _STORE_DTYPES:
        raise ValueError(f"weight_dtype must be one of {_STORE_DTYPES}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    tok_path = CharTokenizer.default_path(model_path)
    if not os.path.exists(tok_path):
        raise FileNotFoundError(
            f"Tokenizer not found at '{tok_path}'. "
            f"The tokenizer JSON must sit beside the model file."
        )

    cfg, ckpt = load_checkpoint(model_path, map_location="cpu")
    with open(tok_path, "r", encoding="utf-8") as f:
        tok_json = json.load(f)
    _ = _load_tok(tok_path)  # fail early on an unreadable tokenizer

    tensors, blobs, offset = [], [], 0
    for k, v in ckpt["state_dict"].items():
        if "causal_mask" in k or not isinstance(v, torch.Tensor):
            continue
        if not v.is_floating_point():
            raise ValueError(
                f"Cannot pack '{k}' with dtype {v.dtype} — "
                f"pack the fp32 original, not a quantized export."
            )
        dt = "f32" if (weight_dtype != "f32" and v.dim() <= 1) else weight_dtype
        blob = _to_blob(v, dt)
        shape = list(v.shape)
        tensors.append({"name": k, "dtype": dt, "shape": shape,
                        "numel": v.numel(), "offset": offset,
                        "nbytes": len(blob)})
        blobs.append(blob)
        offset += len(blob) + _pad_len(len(blob))

    manifest = {
        "format": "scw",
        "version": VERSION,
        "weight_dtype": weight_dtype,
        "name": name or os.path.splitext(os.path.basename(model_path))[0],
        "config": dataclasses.asdict(cfg),
        "tokenizer": tok_json,
        "tensors": tensors,
        "data_size": offset,
        # 64-char placeholder: the real digest is patched in place later,
        # keeping the manifest byte length (and all offsets) fixed.
        "sha256": "0" * 64,
    }
    meta = json.dumps(manifest, ensure_ascii=False).encode("utf-8")

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    h = hashlib.sha256()
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<Q", len(meta)))
        f.write(meta)
        pad = _pad_len(16 + len(meta))
        f.write(b"\x00" * pad)
        for blob in blobs:
            f.write(blob)
            f.write(b"\x00" * _pad_len(len(blob)))
            h.update(blob)
    # The digest covers the tensor region; it is patched into the stored
    # manifest in place (same byte length, so offsets never move).
    _patch_sha256(out_path, h.hexdigest())
    return out_path


def _patch_sha256(path: str, digest: str) -> None:
    """Swap the 64-zero placeholder for the real digest, same length."""
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
    old = ('"sha256": "' + "0" * 64).encode("utf-8")
    new = ('"sha256": "' + digest).encode("utf-8")
    assert len(old) == len(new)
    with open(path, "r+b") as f:
        if f.read(4) != MAGIC:
            raise ValueError(f"Not an .scw file: {path!r}")
        (ver,) = struct.unpack("<I", f.read(4))
        if ver != VERSION:
            raise ValueError(f"Unsupported .scw version {ver}")
        (meta_len,) = struct.unpack("<Q", f.read(8))
        region = bytearray(f.read(meta_len))
        i = region.find(old)
        if i == -1:
            raise RuntimeError("digest placeholder not found in manifest")
        region[i:i + len(old)] = new
        f.seek(16)
        f.write(region)


# ---------------------------------------------------------------------------
# Load (.scw -> config + state dict, mmap zero-copy for f32)
# ---------------------------------------------------------------------------

def _read_manifest(path: str) -> tuple:
    """Return (manifest dict, data_start offset)."""
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError(f"Not an .scw file: {path!r}")
        (ver,) = struct.unpack("<I", f.read(4))
        if ver != VERSION:
            raise ValueError(f"Unsupported .scw version {ver} in {path!r}")
        (meta_len,) = struct.unpack("<Q", f.read(8))
        manifest = json.loads(f.read(meta_len).decode("utf-8"))
    data_start = 16 + meta_len + _pad_len(16 + meta_len)
    return manifest, data_start


def load_scw(path: str, verify: bool = False):
    """
    Load a .scw file: returns (GPTConfig, state_dict, keepalive).

    f32 tensors are zero-copy mmap views (keep `keepalive` referenced as
    long as the model lives — GPT.load does this automatically).

    verify=False by default (same stance as safetensors/GGUF): manifest
    lengths and load-time shape checks catch corruption; pass True to
    additionally check the sha256 (one extra pass over the tensor data).
    """
    from .model import GPTConfig

    manifest, data_start = _read_manifest(path)
    f = open(path, "rb")
    # Copy-on-write: tensors read straight from the mapping, writes (if any)
    # never reach the file.
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
    mv = memoryview(mm)[data_start:data_start + manifest["data_size"]]

    if verify and "sha256" in manifest:
        digest = hashlib.sha256(mv).hexdigest()
        if digest != manifest["sha256"]:
            # mv is the only live export here; release it so close() works.
            mv.release()
            mm.close()
            f.close()
            raise ValueError(f"Checksum mismatch in {path!r}: file is corrupt.")

    state = {}
    for t in manifest["tensors"]:
        raw = mv[t["offset"]:t["offset"] + t["nbytes"]]
        dt, shape = t["dtype"], t["shape"]
        if dt == "f32":
            param = torch.frombuffer(raw, dtype=torch.float32).reshape(shape)
        elif dt == "f16":
            param = torch.frombuffer(raw, dtype=torch.float16).reshape(shape).float()
        elif dt == "q8_0":
            param = dequantize_q8_0(bytes(raw), t["numel"]).reshape(shape)
        else:
            raise ValueError(f"Unknown tensor dtype {dt!r} in {path!r}")
        state[t["name"]] = param

    fields = set(GPTConfig.__dataclass_fields__)
    cfg = GPTConfig(**{k: v for k, v in manifest["config"].items()
                       if k in fields})
    keepalive = (mm, f)  # holds the mapping open; attach to the model
    return cfg, state, keepalive


def load_tokenizer_for(model_path: str):
    """
    Tokenizer for a model path: embedded content for .scw, sidecar JSON
    for .pth. No files are written.
    """
    if model_path.endswith(".scw"):
        manifest, _ = _read_manifest(model_path)
        data = manifest["tokenizer"]
        if isinstance(data, dict) and data.get("type") == "bpe":
            from .bpe import BPETokenizer
            return BPETokenizer.from_dict(data)
        from .tokenizer import CharTokenizer
        return CharTokenizer.from_dict(data)
    from .bpe import load_tokenizer
    from .tokenizer import CharTokenizer
    return load_tokenizer(CharTokenizer.default_path(model_path))


def scw_info(path: str) -> dict:
    """Manifest summary for `scw info` (no tensor data touched)."""
    manifest, _ = _read_manifest(path)
    n_params = sum(t["numel"] for t in manifest["tensors"])
    by_dtype: dict = {}
    for t in manifest["tensors"]:
        by_dtype[t["dtype"]] = by_dtype.get(t["dtype"], 0) + t["nbytes"]
    tok = manifest.get("tokenizer", {})
    return {
        "name": manifest.get("name"),
        "version": manifest.get("version"),
        "weight_dtype": manifest.get("weight_dtype"),
        "arch": manifest.get("config", {}).get("arch", "shimba"),
        "tokenizer": tok.get("type", "char"),
        "vocab_size": tok.get("vocab_size"),
        "params": n_params,
        "tensors": len(manifest["tensors"]),
        "bytes_by_dtype": by_dtype,
        "file_mb": os.path.getsize(path) / (1024 * 1024),
    }


def unpack_model(scw_path: str, out_pth: str) -> tuple:
    """
    Unpack a .scw back to canonical .pth + sidecar tokenizer JSON.
    Returns (pth path, tokenizer path).
    """
    import torch as _torch
    from .checkpoint import FORMAT_VERSION
    from .model import GPT
    from .tokenizer import CharTokenizer

    model = GPT.load(scw_path)  # device stays CPU, buffers rebuilt
    model._scw_keepalive = None  # release the mapping; params own copies now
    for p in model.parameters():
        p.data = p.data.detach().clone()
    out_dir = os.path.dirname(os.path.abspath(out_pth))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    _torch.save({"format_version": FORMAT_VERSION, "config": model.cfg,
                 "state_dict": model.state_dict()}, out_pth)
    tok = load_tokenizer_for(scw_path)
    tok_path = CharTokenizer.default_path(out_pth)
    tok.save(tok_path)
    return out_pth, tok_path
