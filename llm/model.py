"""
model.py — Decoder-only Transformer (CPU-optimised), three architectures.

`arch` selects the block design; everything else (training loop, checkpoint
format, char tokenizer) is shared:

shimba (default) — LayerNorm, learned absolute positions, fused-QKV MHA,
    GELU MLP, no biases.
gpt2             — same block as shimba but with biases on every Linear
    and LayerNorm, matching HF GPT-2.
llama            — RMSNorm, RoPE positions (no position embedding), SwiGLU
    MLP, optional GQA via n_head_kv, no biases.

Memory strategy:
  - float32 throughout (no mixed precision needed for CPU)
  - Weight tying between token embedding and LM head
  - Flash-attention-style manual scaled dot-product (torch.nn.functional)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from .checkpoint import load_checkpoint, save_checkpoint


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    """All hyperparameters for the model in one place."""
    vocab_size: int   = 50257   # overridden by tokenizer at runtime
    block_size: int   = 512     # maximum context length
    n_embd:     int   = 256     # embedding / hidden dimension
    n_head:     int   = 4       # number of attention heads
    n_layer:    int   = 4       # number of transformer blocks
    dropout:    float = 0.1     # dropout probability
    bias:       bool  = False   # use bias in Linear / LayerNorm?
    arch:       str   = "shimba"  # shimba | gpt2 | llama
    n_head_kv:  int | None = None  # GQA key/value heads (llama); None → n_head
    rope_base:  float = 10000.0    # RoPE frequency base (llama)


ARCHES = ("shimba", "gpt2", "llama")


def cfg_arch(cfg: GPTConfig) -> str:
    """Architecture with a fallback for checkpoints saved before it existed."""
    return getattr(cfg, "arch", "shimba") or "shimba"


def cfg_n_kv(cfg: GPTConfig) -> int:
    """Key/value head count (GQA); plain MHA when equal to n_head."""
    return getattr(cfg, "n_head_kv", None) or cfg.n_head


def uses_rope(cfg: GPTConfig) -> bool:
    return cfg_arch(cfg) == "llama"


def uses_rmsnorm(cfg: GPTConfig) -> bool:
    return cfg_arch(cfg) == "llama"


def uses_swiglu(cfg: GPTConfig) -> bool:
    return cfg_arch(cfg) == "llama"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch built-in doesn't expose bias=False)."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class RMSNorm(nn.Module):
    """Root-mean-square norm, no bias, no mean-centering (llama arch)."""

    def __init__(self, ndim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + 1e-5)
        return (self.weight * x.to(self.weight.dtype)).to(x.dtype)


def make_norm(ndim: int, cfg: GPTConfig) -> nn.Module:
    return RMSNorm(ndim) if uses_rmsnorm(cfg) else LayerNorm(ndim, cfg.bias)


def _rope_cos_sin(seq_len: int, head_dim: int, base: float,
                  device: torch.device) -> tuple:
    """RoPE rotation angles for positions 0..seq_len-1 (NeoX half-split)."""
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float()
                          / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)                       # (T, hd/2)
    return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate q/k of shape (..., T, head_dim) in place-free fashion."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (decoder) self-attention.

    Uses PyTorch's scaled_dot_product_attention when available (≥2.0)
    which is both memory-efficient and fast even on CPU. With n_head_kv <
    n_head the key/value heads are repeated (GQA); with the llama arch the
    queries and keys get RoPE rotations.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, \
            f"n_embd ({cfg.n_embd}) must be divisible by n_head ({cfg.n_head})"
        n_kv = cfg_n_kv(cfg)
        assert cfg.n_head % n_kv == 0, \
            f"n_head ({cfg.n_head}) must be divisible by n_head_kv ({n_kv})"
        if uses_rope(cfg):
            assert (cfg.n_embd // cfg.n_head) % 2 == 0, \
                "RoPE needs an even head_dim"

        self.n_head   = cfg.n_head
        self.n_kv     = n_kv
        self.n_embd   = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout  = cfg.dropout
        self.rope     = uses_rope(cfg)
        self.rope_base = getattr(cfg, "rope_base", 10000.0) or 10000.0

        # Single fused projection for Q, K, V
        self.c_attn = nn.Linear(cfg.n_embd, (cfg.n_head + 2 * n_kv) * self.head_dim,
                                bias=cfg.bias)
        # Output projection
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Causal mask — registered as buffer (not a parameter)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
            .view(1, 1, cfg.block_size, cfg.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape   # batch, time (seq len), channels

        # Compute Q, K, V in one matmul then split
        qkv = self.c_attn(x)
        q, k, v = qkv.split((self.n_head * self.head_dim,
                             self.n_kv * self.head_dim,
                             self.n_kv * self.head_dim), dim=2)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv, self.head_dim).transpose(1, 2)

        if self.rope:
            cos, sin = _rope_cos_sin(T, self.head_dim, self.rope_base, x.device)
            q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)

        if self.n_kv != self.n_head:
            # Grouped-query attention: repeat each kv head across its group.
            rep = self.n_head // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Try efficient SDPA (PyTorch ≥ 2.0); fall back to manual
        try:
            # is_causal=True automatically applies causal mask
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        except TypeError:
            # Older PyTorch — manual scaled dot-product
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale           # (B, nh, T, T)
            att = att.masked_fill(
                self.causal_mask[:, :, :T, :T] == 0, float('-inf')
            )
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v                                        # (B, nh, T, hd)

        # Reassemble heads → (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    """
    Position-wise feed-forward network.

    GELU variant: expands to 4× hidden dim with GELU activation.
    SwiGLU variant (llama arch): SiLU-gated pair with the same 4× width.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = 4 * cfg.n_embd
        self.swiglu = uses_swiglu(cfg)
        if self.swiglu:
            self.gate = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
            self.up   = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
            self.proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        else:
            self.fc   = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
            self.proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.swiglu:
            return self.drop(self.proj(F.silu(self.gate(x)) * self.up(x)))
        return self.drop(self.proj(self.act(self.fc(x))))


class TransformerBlock(nn.Module):
    """One decoder block: pre-norm self-attention + pre-norm MLP."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1  = make_norm(cfg.n_embd, cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = make_norm(cfg.n_embd, cfg)
        self.mlp  = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # attention residual
        x = x + self.mlp(self.ln2(x))    # MLP residual
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    Shimba GPT — decoder-only transformer language model.

    Key design choices for CPU efficiency:
      * Weight tying: lm_head shares weights with token embedding
        (saves ~vocab_size × n_embd × 4 bytes of RAM).
      * All operations in float32 — no device transfers needed.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.rope = uses_rope(cfg)

        modules = dict(
            wte  = nn.Embedding(cfg.vocab_size, cfg.n_embd),   # token embeddings
            drop = nn.Dropout(cfg.dropout),
            h    = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)]),
            ln_f = make_norm(cfg.n_embd, cfg),
        )
        if not self.rope:
            # Learned absolute positions (shimba/gpt2); RoPE needs none.
            modules["wpe"] = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.transformer = nn.ModuleDict(modules)

        # LM head — no bias, weights tied to token embedding below
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight   # weight tying

        # Initialise weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/√(2 × n_layer) as in GPT-2 paper
        for name, p in self.named_parameters():
            if name.endswith(("c_proj.weight", "proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[model] parameters: {n_params:,}  "
              f"(~{n_params * 4 / 1024**2:.1f} MB float32)")

    # ------------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,              # (B, T) integer token ids
        targets: torch.Tensor = None,   # (B, T) shifted targets for training
    ):
        B, T = idx.shape
        assert T <= self.cfg.block_size, \
            f"Sequence length {T} exceeds block_size {self.cfg.block_size}"

        # Token embeddings, plus learned positions unless RoPE handles it.
        tok_emb = self.transformer.wte(idx)                # (B, T, n_embd)
        if self.rope:
            x = self.transformer.drop(tok_emb)
        else:
            pos = torch.arange(T, device=idx.device)       # (T,)
            pos_emb = self.transformer.wpe(pos)            # (T, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training: compute loss over all positions
            logits = self.lm_head(x)                       # (B, T, V)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Inference: only compute logits for last token (saves memory)
            logits = self.lm_head(x[:, [-1], :])           # (B, 1, V)
            loss = None

        return logits, loss

    # ------------------------------------------------------------------
    def configure_optimizer(
        self,
        lr: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        """
        AdamW with weight decay applied only to 2-D parameters
        (weight matrices) — not to biases, LayerNorm scales, embeddings.
        """
        decay, no_decay = set(), set()
        whitelist = (nn.Linear,)
        blacklist = (LayerNorm, RMSNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)

        # lm_head.weight is tied → already in decay via wte; remove duplicate
        decay.discard("lm_head.weight")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)],    "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95))

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Save model weights + config to a single .pth file.

        Safe to call on a torch.compile'd module and from GPU: the wrapper is
        peeled off and tensors are moved to CPU before writing.
        """
        save_checkpoint(path, self)
        print(f"[model] saved → {path}")

    @classmethod
    def load(cls, path: str, device: "torch.device | str | None" = None) -> "GPT":
        """
        Load model from a .pth or .scw file.

        .pth tolerates checkpoints written by older versions of this project:
        weights stored under "model" or "weights", and keys prefixed with
        `_orig_mod.` by torch.compile. .scw is the single-file format
        (weights + config + tokenizer); its tensors stay memory-mapped.
        """
        from .device import resolve_device

        if path.endswith(".scw"):
            from .scw import load_scw

            cfg, state, keepalive = load_scw(path)
            model = cls(cfg)
            # Buffers like causal_mask are rebuilt by __init__ and not
            # stored in the file, so anything else missing is an error.
            incompatible = model.load_state_dict(state, strict=False)
            missing = [k for k in incompatible.missing_keys
                       if "causal_mask" not in k]
            if missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Error(s) in loading {path}: missing={missing} "
                    f"unexpected={incompatible.unexpected_keys}")
            model.eval()
            # f32 tensors arrived as mmap views; point the parameters at
            # them so the model shares RAM with the file (f16/q8_0 already
            # came back as converted copies and stay as loaded).
            own = dict(model.named_parameters())
            for k, v in state.items():
                p = own.get(k)
                if (p is not None and v.dtype == torch.float32
                        and v.device.type == "cpu" and v.is_contiguous()
                        and v.shape == p.shape and v.numel() == p.numel()):
                    p.data = v
            # Keeps the mmap alive as long as the model does.
            model._scw_keepalive = keepalive
        else:
            cfg, checkpoint = load_checkpoint(path, map_location="cpu")
            model = cls(cfg)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()

        if device is not None:
            model.to(resolve_device(device))

        print(f"[model] loaded ← {path}")
        return model
