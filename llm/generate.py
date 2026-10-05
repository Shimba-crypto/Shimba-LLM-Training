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
    Generation stops when the decoded continuation ends with one of these
    strings, and the matched suffix is stripped from the returned text.
    Defaults to the corpus document separator, because a model trained on
    joined documents learns to emit that separator between answers and
    would otherwise print it after every reply. Pass `stop_strings=[]` to
    disable and see raw output.

Thinking traces
---------------
Some instruction datasets wrap reasoning in tags
(e.g. `<|begin_of_thought|>...<|end_of_thought|>`). `split_thinking`
separates that span from the final answer for display.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F

from .corpus import DOC_SEPARATOR
from .model import GPT
from .tokenizer import CharTokenizer

# The separator is written as "\n\n====\n\n". A model reproduces it after
# finishing an answer, so only the leading newline-plus-equals run is needed
# to recognise it early and stop before the whole 60 characters are emitted.
_SEP_STOP = "=" * 8


def _default_stop_strings() -> list:
    """Stop at the corpus separator unless the caller opts out."""
    return [DOC_SEPARATOR.strip(), _SEP_STOP, "\nInstruction:", "\nQuestion:"]


def _match_stop(text: str, stops: list) -> Optional[str]:
    """Return the longest stop string `text` ends with, or None."""
    hit = None
    for s in stops:
        if s and text.endswith(s) and (hit is None or len(s) > len(hit)):
            hit = s
    return hit


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
    stop_strings:       Optional[List[str]] = None,
) -> str:
    """
    Generate `max_new_tokens` tokens given a text `prompt`.

    Returns the full string (prompt + generated continuation), with any
    matched stop string removed.

    `stop_strings=None` stops at the corpus document separator and at a new
    `Instruction:` header, which is what keeps a model trained on joined
    documents from printing the separator after every reply. Pass `[]` to
    see raw output.

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

    stops = _default_stop_strings() if stop_strings is None else [
        s for s in stop_strings if s]
    # Only the tail can match a stop string, so a fixed window is enough and
    # keeps the per-token cost flat no matter how long the reply gets.
    tail_window = max((len(s) for s in stops), default=0) + 8

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

        if stops:
            tail = tokenizer.decode((ids + generated_ids)[-tail_window:])
            if _match_stop(tail, stops) is not None:
                break

    text = tokenizer.decode(ids + generated_ids)
    if stops:
        # Trim at the earliest stop string anywhere in the continuation.
        # Working on decoded text rather than tokens keeps this correct for
        # BPE, where one character can span several tokens.
        cont = tokenizer.decode(generated_ids)
        cut = min((cont.find(s) for s in stops if cont.find(s) != -1),
                  default=-1)
        if cut != -1:
            cont = cont[:cut]
            return tokenizer.decode(ids) + cont
    return text


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
