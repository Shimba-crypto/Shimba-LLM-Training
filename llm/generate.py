"""
generate.py — Autoregressive text generation for the Shimba LLM.

Sampling strategies
-------------------
temperature : float (default 1.0)
    < 1.0  → sharper / more deterministic (greedy at → 0)
    > 1.0  → more random / creative
    = 0.0  → pure greedy (argmax)

top_k : int (default 0 = disabled)
    Keep only the top-k highest-probability tokens before sampling.
    Values in the range 20–200 work well in practice.

top_p : float (default 1.0 = disabled)
    Nucleus sampling — keep the smallest set of tokens whose cumulative
    probability ≥ top_p. Combine with temperature for best results.

repetition_penalty : float (default 1.0 = disabled)
    Divide logits of already-seen tokens by this factor (> 1.0 penalises
    repetition, < 1.0 encourages it).

stop_tokens : list[int]
    Generation stops early if any of these token ids is produced.
"""

from typing import List, Optional
import torch
import torch.nn.functional as F

from .model import GPT
from .tokenizer import CharTokenizer


# ---------------------------------------------------------------------------
# Core sampling function
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model:              GPT,
    tokenizer:          CharTokenizer,
    prompt:             str,
    max_new_tokens:     int           = 200,
    temperature:        float         = 0.8,
    top_k:              int           = 40,
    top_p:              float         = 0.95,
    repetition_penalty: float         = 1.1,
    stop_tokens:        Optional[List[int]] = None,
) -> str:
    """
    Generate `max_new_tokens` tokens given a text `prompt`.

    Returns the full string (prompt + generated continuation).

    Works on whatever device the model lives on — the token tensor is created
    on that device rather than assumed to be CPU.
    """
    model.eval()
    block_size = model.cfg.block_size

    # Tokens must live on the same device as the model's weights, or the
    # embedding lookup fails on CUDA/MPS.
    device = next(model.parameters()).device

    # Encode prompt
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [0]   # fall back to PAD if prompt is empty
    idx = torch.tensor([ids], dtype=torch.int64, device=device)  # (1, T)

    stop_set = set(stop_tokens) if stop_tokens else set()
    generated_ids: List[int] = []

    for _ in range(max_new_tokens):
        # Crop context to block_size
        idx_cond = idx[:, -block_size:]

        # Forward pass — returns logits for the last position only
        logits, _ = model(idx_cond)   # (1, 1, V)
        logits = logits[:, -1, :]     # (1, V)

        # ── Repetition penalty ──────────────────────────────────────
        if repetition_penalty != 1.0 and generated_ids:
            seen = set(ids + generated_ids)
            for token_id in seen:
                if token_id < logits.size(-1):
                    logits[0, token_id] /= repetition_penalty

        # ── Temperature ─────────────────────────────────────────────
        if temperature == 0.0:
            # Greedy
            next_id = logits.argmax(dim=-1, keepdim=True)  # (1, 1)
        else:
            logits = logits / temperature

            # ── Top-k ──────────────────────────────────────────────
            if top_k > 0:
                top_k_clamped = min(top_k, logits.size(-1))
                topk_vals = torch.topk(logits, top_k_clamped).values
                threshold = topk_vals[:, -1].unsqueeze(-1)   # kth value
                logits = logits.masked_fill(logits < threshold, float('-inf'))

            # ── Top-p (nucleus) ────────────────────────────────────
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Remove tokens with cumulative prob above top_p
                remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove_mask] = float('-inf')
                # Scatter back to original order
                logits = torch.scatter(logits, 1, sorted_idx, sorted_logits)

            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

        token_int = next_id.item()

        # Stop if requested
        if token_int in stop_set:
            break

        generated_ids.append(token_int)
        idx = torch.cat([idx, next_id], dim=1)  # grow sequence

    full_ids = ids + generated_ids
    return tokenizer.decode(full_ids)


# ---------------------------------------------------------------------------
# Convenience: stream generation to stdout
# ---------------------------------------------------------------------------

@torch.no_grad()
def stream_generate(
    model:          GPT,
    tokenizer:      CharTokenizer,
    prompt:         str,
    max_new_tokens: int   = 200,
    temperature:    float = 0.8,
    top_k:          int   = 40,
    top_p:          float = 0.95,
) -> None:
    """
    Same as `generate` but prints each token as it is produced.
    Useful for interactive demos.
    """
    import sys
    model.eval()
    block_size = model.cfg.block_size
    device = next(model.parameters()).device

    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [0]

    # Print the prompt first
    sys.stdout.write(prompt)
    sys.stdout.flush()

    idx = torch.tensor([ids], dtype=torch.int64, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        if top_k > 0:
            top_k_clamped = min(top_k, logits.size(-1))
            topk_vals = torch.topk(logits, top_k_clamped).values
            logits = logits.masked_fill(logits < topk_vals[:, -1:], float('-inf'))

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove] = float('-inf')
            logits = torch.scatter(logits, 1, sorted_idx, sorted_logits)

        probs   = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        char_str = tokenizer.decode([next_id.item()])
        sys.stdout.write(char_str)
        sys.stdout.flush()

        idx = torch.cat([idx, next_id], dim=1)

    sys.stdout.write("\n")
    sys.stdout.flush()
