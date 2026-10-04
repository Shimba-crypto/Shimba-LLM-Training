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

stop_strings : list[str]
    Generation stops when the decoded text ends with one of these strings
    (e.g. an end-of-answer tag). `generate` strips the matched suffix from
    the returned text; the streaming variant stops without retracting
    already-printed chunks.

Thinking traces
---------------
Some instruction datasets wrap reasoning in tags
(e.g. `<|begin_of_thought|>...<|end_of_thought|>`). `split_thinking`
separates that span from the final answer for display.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F

from .model import GPT
from .tokenizer import CharTokenizer


def split_thinking(text: str, think_start: Optional[str] = None,
                   think_end: Optional[str] = None) -> Tuple[Optional[str], str]:
    """
    Split generated text into (thinking, answer) at think tags.

    Returns (None, text) when think_end is unset or absent. When present,
    thinking is the span before think_end (after think_start, if given and
    found) and answer is everything after it; both stripped.
    """
    if not think_end or think_end not in text:
        return None, text
    head, _, tail = text.partition(think_end)
    if think_start:
        i = head.find(think_start)
        thinking = head[i + len(think_start):] if i != -1 else head
    else:
        thinking = head
    return thinking.strip(), tail.strip()


# ---------------------------------------------------------------------------
# Core sampling function
# ---------------------------------------------------------------------------

def _apply_repetition_penalty(logits: torch.Tensor, seen_ids, penalty: float) -> None:
    """Divide logits of seen tokens by penalty, in one indexed op."""
    if penalty == 1.0 or not seen_ids:
        return
    # seen_ids is a set/list of ints; single indexed division.
    idx = torch.as_tensor(list(seen_ids), dtype=torch.long, device=logits.device)
    idx = idx[idx < logits.size(-1)]
    if idx.numel():
        logits[0, idx] /= penalty


def _sample_next(logits: torch.Tensor, temperature: float,
                 top_k: int, top_p: float) -> torch.Tensor:
    """Apply temperature / top-k / top-p and sample one token id."""
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)  # (1, 1)
    logits = logits / temperature

    if top_k > 0:
        top_k_clamped = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k_clamped).values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < threshold, float('-inf'))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs_sorted = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs_sorted, dim=-1)
        remove_mask = cumulative - probs_sorted > top_p
        sorted_logits[remove_mask] = float('-inf')
        logits = torch.scatter(logits, 1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)  # (1, 1)


@torch.inference_mode()
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
    seen_ids = set(ids)

    for _ in range(max_new_tokens):
        # Only the last block_size tokens affect the output; keep idx bounded
        # so long sessions do not grow memory and copy cost per step.
        if idx.size(1) > block_size:
            idx = idx[:, -block_size:]
        idx_cond = idx[:, -block_size:]

        # Forward pass — returns logits for the last position only
        logits, _ = model(idx_cond)   # (1, 1, V)
        logits = logits[:, -1, :]     # (1, V)

        _apply_repetition_penalty(logits, seen_ids, repetition_penalty)
        next_id = _sample_next(logits, temperature, top_k, top_p)

        token_int = next_id.item()

        # Stop if requested
        if token_int in stop_set:
            break

        generated_ids.append(token_int)
        seen_ids.add(token_int)
        idx = torch.cat([idx, next_id], dim=1)  # grow sequence

    full_ids = ids + generated_ids
    return tokenizer.decode(full_ids)


def iter_generate(
    model:              GPT,
    tokenizer:          CharTokenizer,
    prompt:             str,
    max_new_tokens:     int           = 200,
    temperature:        float         = 0.8,
    top_k:              int           = 40,
    top_p:              float         = 0.95,
    repetition_penalty: float         = 1.1,
    stop_tokens:        Optional[List[int]] = None,
):
    """
    Yield decoded text chunks as they are generated.

    Same sampling as `generate`, but yields after each token so callers can
    display output incrementally.
    """
    model.eval()
    block_size = model.cfg.block_size
    device = next(model.parameters()).device

    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [0]
    idx = torch.tensor([ids], dtype=torch.int64, device=device)

    stop_set = set(stop_tokens) if stop_tokens else set()
    seen_ids = set(ids)
    out_ids: List[int] = []

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if idx.size(1) > block_size:
                idx = idx[:, -block_size:]
            idx_cond = idx[:, -block_size:]

            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]

            _apply_repetition_penalty(logits, seen_ids, repetition_penalty)
            next_id = _sample_next(logits, temperature, top_k, top_p)
            token_int = next_id.item()
            if token_int in stop_set:
                break
            out_ids.append(token_int)
            seen_ids.add(token_int)
            idx = torch.cat([idx, next_id], dim=1)
            yield tokenizer.decode([token_int])


# ---------------------------------------------------------------------------
# Convenience: stream generation to stdout
# ---------------------------------------------------------------------------

@torch.inference_mode()
def stream_generate(
    model:          GPT,
    tokenizer:      CharTokenizer,
    prompt:         str,
    max_new_tokens: int   = 200,
    temperature:    float = 0.8,
    top_k:          int   = 40,
    top_p:          float = 0.95,
    repetition_penalty: float = 1.1,
    stop_tokens:    Optional[List[int]] = None,
) -> None:
    """
    Same as `generate` but prints each token as it is produced.
    Useful for interactive demos.
    """
    import sys
    model.eval()

    # Print the prompt first
    sys.stdout.write(prompt)
    sys.stdout.flush()

    for chunk in iter_generate(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_tokens=stop_tokens,
    ):
        sys.stdout.write(chunk)
        sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()
