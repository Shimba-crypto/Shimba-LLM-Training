"""
sft.py — Supervised fine-tuning for Shimba LLM models.

Pretraining computes loss over every token, prompts included. SFT trains on
the *response* tokens only: prompt positions get label -1, which the model's
cross-entropy already ignores (ignore_index=-1). Two consequences:

  * The model stops spending capacity memorising your questions and spends
    it on answers — the standard SFT behaviour.
  * Validation is split by document, not by token window, so near-duplicate
    windows no longer leak from train into val and flatter the loss.

Data
----
JSON/JSONL objects are read as (prompt, response) pairs, mirroring the
field priority in llm/corpus.py but keeping the boundary the corpus loader
throws away: instruction/output, prompt/completion, question/answer,
user/assistant, input/output, message lists, or a bare text/content field
(which trains whole, prompt empty). Plain .txt files train whole the same
way. Templates render pairs exactly like corpus.py does, so --sft on the
same files sees the same text, only masked.

Templates
---------
instruction (default): prompt="Instruction: {q}\\nResponse: ", response=out
chat:                  prompt="User: {q}\\nAssistant: ",      response=out
"""

import json
import os
import random
import sys

import torch

from .corpus import DOC_SEPARATOR, collect_files, detect_format
from .data import DataLoader  # noqa: F401 (re-exported for SFT callers)
from .train import Trainer

IGNORE = -1


def render_pair(instruction: str, output: str,
                template: str = "instruction") -> tuple:
    """
    Split one pair into (prompt_text, response_text).

    Empty instructions (plain .txt / bare text fields) yield an empty
    prompt, meaning every token trains — the SFT equivalent of pretraining
    on that document.
    """
    q = (instruction or "").strip()
    a = (output or "").strip()
    if template == "chat":
        return (f"User: {q}\nAssistant: " if q else "", a)
    return (f"Instruction: {q}\nResponse: " if q else "", a)


def encode_pair(tokenizer, prompt_text: str, response_text: str,
                block_size: int):
    """
    Encode one pair into (x, y) int lists of exactly block_size.

    Labels mask the prompt with -1; padding (id 0 in, -1 out) fills short
    rows. A prompt-free document is chunked into whole windows so no token
    is lost. Returns a list of items — usually one, several for long
    prompt-free texts, none when there is nothing trainable.
    """
    p = tokenizer.encode(prompt_text) if prompt_text else []
    r = tokenizer.encode(response_text) if response_text else []
    if not r and not p:
        return []
    if not p:
        # No boundary to preserve: plain next-token training on windows.
        ids = r
        items = []
        for s in range(0, len(ids), block_size):
            chunk = ids[s:s + block_size + 1]
            if len(chunk) < 2:
                continue
            x = chunk[:-1]
            y = chunk[1:]
            pad = block_size - len(x)
            items.append((x + [0] * pad, y + [IGNORE] * pad))
        return items
    # Keep at least the tail of the response; trim a long prompt from the
    # left (early context matters least for the answer).
    max_p = block_size - 1
    if len(p) > max_p:
        p = p[-max_p:]
    room = block_size - len(p)
    r = r[:max(1, room)]
    if not r:
        return []  # prompt-only row: nothing trainable under masking
    # NOTE: prompt_len comes from a separate encode() call, so for BPE the
    # boundary can sit a token or two off where a joint encode would merge
    # across the junction. Exact for char tokenizers; negligible either way.
    full = (p + r)[:block_size + 1]
    x = full[:-1]
    y = full[1:]
    # Mask every label position that predicts a prompt token, i.e. the
    # target index sits inside the prompt.
    y = [t if (j + 1) >= len(p) else IGNORE for j, t in enumerate(y)]
    pad = block_size - len(x)
    return [(x + [0] * pad, y + [IGNORE] * pad)]


class SFTDataset:
    """
    Pre-encoded (x, y) pairs, each exactly block_size long.

    Duck-typing matches TextDataset (__len__/__getitem__), so the stock
    DataLoader and Trainer consume it unchanged.
    """

    def __init__(self, items: list):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        x, y = self.items[idx]
        return (torch.tensor(x, dtype=torch.int64),
                torch.tensor(y, dtype=torch.int64))


def make_sft_splits(pairs: list, tokenizer, block_size: int,
                    val_fraction: float = 0.1, template: str = "instruction",
                    seed: int = 42):
    """
    Shuffle pairs by document, split, encode with prompt masking.

    Returns (train_ds, val_ds, stats) where stats reports pair counts,
    dropped empties, and the fraction of labels that actually train —
    the number that tells you whether --sft is doing anything.
    """
    rng = random.Random(seed)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    cut = int(len(pairs) * (1 - val_fraction))
    train_idx, val_idx = set(order[:cut]), set(order[cut:])

    out = {}
    dropped = 0
    for name, idxs in (("train", train_idx), ("val", val_idx)):
        items = []
        for i in sorted(idxs):
            prompt_t, response_t = render_pair(*pairs[i], template)
            got = encode_pair(tokenizer, prompt_t, response_t, block_size)
            if got:
                items.extend(got)
            else:
                dropped += 1
        if not items:
            print(f"[error] SFT {name} split is empty — no trainable pairs. "
                  f"Check that the data has response text.")
            sys.exit(1)
        out[name] = SFTDataset(items)

    def trained_frac(ds):
        tot = sum(len(y) for _, y in ds.items)
        masked = sum(1 for _, y in ds.items for t in y if t == IGNORE)
        return 1 - masked / max(1, tot)

    stats = {
        "train_pairs": len(train_idx),
        "val_pairs": len(val_idx),
        "dropped": dropped,
        "train_trained_frac": trained_frac(out["train"]),
    }
    print(f"[sft] {stats['train_pairs']:,} train pairs / "
          f"{stats['val_pairs']:,} val pairs "
          f"({stats['train_trained_frac'] * 100:.0f}% of labels train, "
          f"{dropped} empty dropped)")
    return out["train"], out["val"], stats


class SFTTrainer(Trainer):
    """
    Trainer running on masked SFT datasets.

    No loop changes needed: the stock loop, eval, checkpointing and resume
    all operate on (x, y) batches, and the -1 labels flow through the
    model's existing ignore_index. This subclass exists to name the mode
    in logs and to carry SFT stats alongside the run.
    """

    def __init__(self, model_cfg, train_cfg, train_ds, val_ds, stats=None):
        super().__init__(model_cfg, train_cfg, train_ds, val_ds)
        self.sft_stats = stats or {}
        frac = self.sft_stats.get("train_trained_frac")
        if frac is not None:
            print(f"[sft] response-only loss on {frac * 100:.0f}% of tokens")


# ---------------------------------------------------------------------------
# Pair loading (boundary-preserving variant of llm/corpus.py)
# ---------------------------------------------------------------------------

_MESSAGE_KEYS = ("messages", "conversation", "conversations", "turns")
_CONTENT_KEYS = ("content", "text", "value", "utterance")
_ROLE_KEYS = ("role", "from", "speaker", "author")


def _is_user_turn(turn: dict) -> bool:
    for key in _ROLE_KEYS:
        role = turn.get(key)
        if isinstance(role, str) and role.strip().lower() in (
                "user", "human", "instruction", "prompt", "question"):
            return True
    return False


def _turn_text(turn: dict):
    for key in _CONTENT_KEYS:
        value = turn.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _pair_from_object(obj: dict, text_field=None):
    """Extract (instruction, response) from one JSON object, or None."""
    if not isinstance(obj, dict):
        return None
    if text_field:
        names = [f.strip() for f in text_field.split(",") if f.strip()]
        vals = [obj[n] for n in names
                if isinstance(obj.get(n), str) and obj[n].strip()]
        if vals:
            return ("", "\n".join(v.strip() for v in vals))
        return None
    for key in ("text", "content"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return ("", value.strip())
    if isinstance(obj.get("instruction"), str) and obj["instruction"].strip():
        parts = [obj["instruction"].strip()]
        out = None
        for key in ("input", "context"):
            if isinstance(obj.get(key), str) and obj[key].strip():
                parts.append(obj[key].strip())
        for key in ("output", "completion", "answer", "response"):
            if isinstance(obj.get(key), str) and obj[key].strip():
                out = obj[key].strip()
                break
        if out:
            return ("\n".join(parts), out)
    for a, b in (("prompt", "completion"), ("prompt", "output"),
                 ("question", "answer"), ("input", "output")):
        if isinstance(obj.get(a), str) and obj[a].strip() \
                and isinstance(obj.get(b), str) and obj[b].strip():
            return (obj[a].strip(), obj[b].strip())
    if isinstance(obj.get("user"), str) and isinstance(obj.get("assistant"), str) \
            and obj["user"].strip() and obj["assistant"].strip():
        return (obj["user"].strip(), obj["assistant"].strip())
    for key in _MESSAGE_KEYS:
        turns = obj.get(key)
        if isinstance(turns, list) and turns:
            texts = []
            for t in turns:
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
                elif isinstance(t, dict):
                    c = _turn_text(t)
                    if c:
                        texts.append(c)
            if len(texts) >= 2:
                # All but the last turn is context; the model learns the finale.
                return ("\n".join(texts[:-1]), texts[-1])
            if texts:
                return ("", texts[0])
    strings = [v.strip() for _, v in sorted(obj.items())
               if isinstance(v, str) and v.strip()]
    if strings:
        return ("", "\n".join(strings))
    return None


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def _pairs_from_file(path: str, text_field=None,
                     data_format: str = "auto") -> list:
    fmt = detect_format(path, data_format)
    if fmt == "txt":
        text = _read_text_file(path)
        return [("", text)] if text.strip() else []
    if fmt == "json":
        try:
            payload = json.loads(_read_text_file(path))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"[error] Cannot read '{path}': {exc}")
            sys.exit(1)
        items = payload if isinstance(payload, list) else [payload]
        if len(items) == 1 and isinstance(items[0], dict):
            for value in items[0].values():
                if isinstance(value, list) and value \
                        and all(isinstance(v, (dict, str)) for v in value):
                    items = value
                    break
        pairs = []
        for item in items:
            if isinstance(item, str) and item.strip():
                pairs.append(("", item.strip()))
            elif isinstance(item, dict):
                pair = _pair_from_object(item, text_field)
                if pair and (pair[0].strip() or pair[1].strip()):
                    pairs.append(pair)
        return pairs
    # jsonl: one value per line; bad lines stay as plain text.
    pairs = []
    for line in _read_text_file(path).splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            pairs.append(("", line))
            continue
        if isinstance(obj, str) and obj.strip():
            pairs.append(("", obj.strip()))
        elif isinstance(obj, dict):
            pair = _pair_from_object(obj, text_field)
            if pair and (pair[0].strip() or pair[1].strip()):
                pairs.append(pair)
    return pairs


def load_sft_pairs(data_path: str, pattern: str = "*.txt,*.json,*.jsonl",
                   recurse: bool = True, text_field=None,
                   data_format: str = "auto") -> list:
    """
    Collect (instruction, response) pairs from a file or folder of files.

    Same path/pattern/recurse/format handling as the corpus loader, but
    document boundaries survive: each pair trains with its prompt masked.
    """
    if os.path.isfile(data_path):
        pairs = _pairs_from_file(data_path, text_field, data_format)
        if not pairs:
            print(f"[error] No training pairs found in '{data_path}'")
            sys.exit(1)
        print(f"[sft] {len(pairs)} pair(s) from '{data_path}'")
        return pairs
    if os.path.isdir(data_path):
        files = collect_files(data_path, pattern, recurse)
        if not files:
            print(f"[error] No files matching '{pattern}' found in '{data_path}'")
            sys.exit(1)
        pairs = []
        for fpath in files:
            pairs.extend(_pairs_from_file(fpath, text_field, data_format))
        if not pairs:
            print(f"[error] No training pairs found in '{data_path}'")
            sys.exit(1)
        print(f"[sft] {len(pairs)} pair(s) across {len(files)} file(s)")
        return pairs
    print(f"[error] --data path does not exist: '{data_path}'")
    sys.exit(1)


def render_corpus_text(pairs: list, template: str = "instruction") -> str:
    """
    Plain-text form of the pairs, for tokenizer vocabulary building.

    Uses the same separator as the corpus loader so a char vocabulary
    built here covers exactly the text training sees.
    """
    docs = []
    for instruction, output in pairs:
        prompt_t, response_t = render_pair(instruction, output, template)
        docs.append(prompt_t + response_t)
    return DOC_SEPARATOR.join(docs)
