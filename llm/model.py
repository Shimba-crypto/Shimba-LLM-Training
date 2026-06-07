"""
model.py — Shimba decoder-only Transformer (CPU-optimised)

Architecture:
  Embedding → N × TransformerBlock → LayerNorm → LM Head

Each TransformerBlock:
  LayerNorm → CausalSelfAttention → residual
  LayerNorm → MLP                 → residual

Memory strategy:
  - float32 throughout (no mixed precision needed for CPU)
  - bias=False by default (fewer parameters, slightly faster)
  - Flash-attention-style manual scaled dot-product (torch.nn.functional)
  - Weight tying between token embedding and LM head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


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


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (decoder) self-attention.

    Uses PyTorch's scaled_dot_product_attention when available (≥2.0)
    which is both memory-efficient and fast even on CPU.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, \
            f"n_embd ({cfg.n_embd}) must be divisible by n_head ({cfg.n_head})"

        self.n_head   = cfg.n_head
        self.n_embd   = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout  = cfg.dropout

        # Single fused projection for Q, K, V
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
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
        qkv = self.c_attn(x)                          # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_dim)
        def _reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = _reshape(q), _reshape(k), _reshape(v)

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
    Expands to 4× hidden dim with GELU activation.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = 4 * cfg.n_embd
        self.fc   = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(self.act(self.fc(x))))


class TransformerBlock(nn.Module):
    """One decoder block: pre-norm self-attention + pre-norm MLP."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1  = LayerNorm(cfg.n_embd, cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = LayerNorm(cfg.n_embd, cfg.bias)
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
      * No positional encoding bias parameter (learned absolute positions).
      * All operations in float32 — no device transfers needed.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(cfg.vocab_size, cfg.n_embd),   # token embeddings
            wpe  = nn.Embedding(cfg.block_size, cfg.n_embd),   # position embeddings
            drop = nn.Dropout(cfg.dropout),
            h    = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)]),
            ln_f = LayerNorm(cfg.n_embd, cfg.bias),
        ))

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

        # Token + position embeddings
        pos = torch.arange(T, device=idx.device)          # (T,)
        tok_emb = self.transformer.wte(idx)                # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)                # (T, n_embd)
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
        blacklist = (LayerNorm, nn.Embedding)

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
        """Save model weights + config to a single .pth file."""
        torch.save({"config": self.cfg, "state_dict": self.state_dict()}, path)
        print(f"[model] saved → {path}")

    @classmethod
    def load(cls, path: str) -> "GPT":
        """Load model from a .pth file produced by save()."""
        # PyTorch 2.6+ requires explicit allowlisting of custom classes
        # when weights_only=True (the new default).  We try the safe path
        # first and fall back gracefully for older versions.
        try:
            import torch.serialization as _ts
            with _ts.safe_globals([GPTConfig]):
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except (AttributeError, TypeError):
            # Older PyTorch without safe_globals / weights_only kwarg
            checkpoint = torch.load(path, map_location="cpu")  # type: ignore[call-arg]
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(f"[model] loaded ← {path}")
        return model
