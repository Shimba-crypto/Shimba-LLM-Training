"""
bpe.py — Byte-level BPE tokenizer, zero dependencies.

GPT-2-style subword tokenizer trained on any corpus:

    * Pre-tokenization splits text with a GPT-2-like pattern (approximated
      with stdlib `re`: \\p{L} becomes "unicode letters", \\p{N} becomes
      unicode digits — identical for Latin/CJK text, close elsewhere).
    * Each piece is byte-encoded (byte 0x20 becomes "Ġ", etc.), so the base
      vocabulary of 256 byte tokens covers every input with no dead ends.
    * The most frequent adjacent pair merges until the target vocab size.

Interface matches CharTokenizer (encode/decode/vocab_size, save/load,
default_path, PAD id 0, UNK id 1), so training, generation, chat and GGUF
export accept either. Saved files carry "type": "bpe" plus the merge list;
use `load_tokenizer()` to open either kind.

Note on Ollama: the same pre-tokenizer pattern and merge ranks are what
llama.cpp applies to a GPT-2-style GGUF, so prompts tokenize identically
there and in `encode()`.
"""

import json
import os
import re
from collections import Counter

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# GPT-2 pre-tokenizer, approximated with stdlib `re`:
#   \p{L} -> [^\W\d_] (unicode letters), \p{N} -> \d (unicode digits),
#   "everything else" -> [^\s\w] plus underscore.
# Matches OpenAI's pattern on Latin/CJK text; exotic scripts may split
# slightly differently, which only costs a little efficiency, never text.
BPE_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[^\W\d_]+"
    r"| ?\d+"
    r"| ?(?:[^\s\w]|_)+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def _bytes_to_unicode() -> dict:
    """Byte value -> unicode char, the standard GPT-2 mapping."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_TO_UNICODE = _bytes_to_unicode()
UNICODE_TO_BYTE = {c: b for b, c in BYTE_TO_UNICODE.items()}


def byte_encode(text: str) -> str:
    """Raw text -> GPT-2 byte-encoded string (space becomes "Ġ", ...)."""
    return "".join(BYTE_TO_UNICODE[b] for b in text.encode("utf-8"))


def byte_decode(text: str) -> str:
    """GPT-2 byte-encoded string -> raw text (never raises)."""
    raw = bytearray()
    for ch in text:
        b = UNICODE_TO_BYTE.get(ch)
        if b is not None:
            raw.append(b)
    return raw.decode("utf-8", errors="replace")


def pre_tokenize(text: str) -> list:
    """Split text into BPE pre-tokens (byte-encoded)."""
    return [byte_encode(m.group(0))
            for m in BPE_PATTERN.finditer(text) if m.group(0)]


def get_pairs(symbols: list) -> set:
    """Adjacent symbol pairs in one pre-token split."""
    return {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}


def train_bpe(text: str, vocab_size: int, min_freq: int = 2,
              progress_every: int = 250) -> tuple:
    """
    Learn a BPE merge list from raw text.

    Returns (merges, vocab) where merges is [(left, right), ...] in rank
    order and vocab maps token string -> id: 0 <PAD>, 1 <UNK>, 2..257 the
    256 byte tokens, then one id per merge in order.
    """
    if vocab_size < 260:
        raise ValueError(f"--vocab-size must be at least 260 (got {vocab_size})")

    # Frequency table over pre-tokens; merging works on unique pieces only.
    words = Counter(pre_tokenize(text))
    splits = {w: list(w) for w in words}

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for b in range(256):
        vocab[BYTE_TO_UNICODE[b]] = 2 + b

    # Inverted index: pair -> set of words containing it, for cheap updates.
    pair_to_words: dict = {}
    for w, symbols in splits.items():
        for pair in get_pairs(symbols):
            pair_to_words.setdefault(pair, set()).add(w)

    def pair_counts():
        counts = Counter()
        for pair, ws in pair_to_words.items():
            counts[pair] = sum(words[w] for w in ws)
        return counts

    merges = []
    while len(vocab) < vocab_size:
        counts = pair_counts()
        if not counts:
            break
        best, freq = counts.most_common(1)[0]
        if freq < min_freq:
            break
        merged = "".join(best)
        vocab[merged] = len(vocab)
        merges.append(list(best))
        if len(merges) % progress_every == 0:
            print(f"[bpe] merge {len(merges)}: {best[0]!r}+{best[1]!r} "
                  f"(x{freq}, vocab {len(vocab)})")

        # Re-split every affected word, fixing the pair index as we go.
        for w in list(pair_to_words.pop(best, ())):
            symbols = splits[w]
            out, i = [], 0
            while i < len(symbols):
                if (i < len(symbols) - 1
                        and (symbols[i], symbols[i + 1]) == best):
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            # Drop pairs that vanished with the old split ...
            for pair in get_pairs(symbols):
                ws = pair_to_words.get(pair)
                if ws is not None:
                    ws.discard(w)
                    if not ws:
                        del pair_to_words[pair]
            # ... and index the pairs of the new split.
            splits[w] = out
            for pair in get_pairs(out):
                pair_to_words.setdefault(pair, set()).add(w)

    print(f"[bpe] done: {len(merges)} merges, vocab {len(vocab)}")
    return merges, vocab


class BPETokenizer:
    """
    Byte-level BPE tokenizer with the CharTokenizer interface.

    Special tokens: <PAD> = 0, <UNK> = 1. Bytes alone cover every input,
    so UNK ids should never appear in practice; decode skips specials.
    """

    PAD_TOKEN = PAD_TOKEN
    UNK_TOKEN = UNK_TOKEN
    TYPE = "bpe"

    def __init__(self):
        self.merges: list = []
        self.vocab: dict = {}
        self.char2idx: dict = {}
        self.idx2char: dict = {}
        self.vocab_size: int = 0
        self._ranks: dict = {}
        self._cache: dict = {}
        # Special-token ids. Ours default to PAD=0/UNK=1; an HF import
        # (GPT-2 layout) carries none of those and sets eos_id instead.
        self.pad_id: int | None = 0
        self.unk_id: int | None = 1
        self.eos_id: int | None = None
        self.bos_id: int | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self, text: str, vocab_size: int = 2000,
              min_freq: int = 2) -> "BPETokenizer":
        """Learn merges from raw text and finalize the vocabulary."""
        merges, vocab = train_bpe(text, vocab_size, min_freq)
        self.merges = merges
        self.vocab = vocab
        self._finalize()
        print(f"[tokenizer] BPE vocabulary size: {self.vocab_size} "
              f"(2 special + 256 bytes + {len(merges)} merges)")
        return self

    def _finalize(self) -> None:
        self.char2idx = dict(self.vocab)
        self.idx2char = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self._ranks = {tuple(m): i for i, m in enumerate(self.merges)}
        self._cache = {}

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def _bpe_word(self, word: str) -> tuple:
        """Apply merges (lowest rank first) to one byte-encoded pre-token."""
        if word in self._cache:
            return self._cache[word]
        symbols = list(word)
        if len(symbols) < 2:
            out = tuple(symbols)
            self._cache[word] = out
            return out
        while True:
            pairs = get_pairs(symbols)
            if not pairs:
                break
            ranked = [(self._ranks[p], p) for p in pairs if p in self._ranks]
            if not ranked:
                break
            _, best = min(ranked)
            out, i = [], 0
            while i < len(symbols):
                if (i < len(symbols) - 1
                        and (symbols[i], symbols[i + 1]) == best):
                    out.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
            if len(symbols) == 1:
                break
        out = tuple(symbols)
        self._cache[word] = out
        return out

    def encode(self, text: str) -> list:
        """Convert string to list of integer token ids."""
        unk = self.unk_id if self.unk_id is not None else 1
        ids = []
        for word in pre_tokenize(text):
            for tok in self._bpe_word(word):
                ids.append(self.vocab.get(tok, unk))
        return ids

    def decode(self, ids: list) -> str:
        """Convert list of integer token ids to string."""
        parts = []
        skip = set()
        if self.pad_id is not None:
            skip.add(self.pad_id)
        if self.unk_id is not None:
            skip.add(self.unk_id)
        for i in ids:
            if i in skip:
                continue
            tok = self.idx2char.get(i, UNK_TOKEN)
            if tok not in (PAD_TOKEN, UNK_TOKEN):
                parts.append(tok)
        return byte_decode("".join(parts))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise vocabulary + merges to a JSON file."""
        data = {
            "type": self.TYPE,
            "vocab": self.vocab,
            "merges": self.merges,
            "vocab_size": self.vocab_size,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "eos_id": self.eos_id,
            "bos_id": self.bos_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[tokenizer] saved → {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load a previously saved BPE tokenizer from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, path)

    @classmethod
    def from_dict(cls, data: dict, path: str = "<dict>") -> "BPETokenizer":
        """Build from already-parsed JSON content (files and .scw alike)."""
        tok = cls()
        tok.merges = [list(m) for m in data["merges"]]
        tok.vocab = {k: int(v) for k, v in data["vocab"].items()}
        tok.pad_id = data.get("pad_id", 0)
        tok.unk_id = data.get("unk_id", 1)
        tok.eos_id = data.get("eos_id")
        tok.bos_id = data.get("bos_id")
        tok._finalize()
        print(f"[tokenizer] loaded ← {path}  (BPE vocab_size={tok.vocab_size})")
        return tok

    @staticmethod
    def default_path(model_path: str) -> str:
        """Derive tokenizer JSON path from model .pth path."""
        base, _ = os.path.splitext(model_path)
        return base + "_tokenizer.json"


def load_tokenizer(path: str):
    """
    Open a tokenizer JSON of either kind (char or BPE), detected from the
    stored "type" field. Files written before the field existed are char.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and data.get("type") == BPETokenizer.TYPE:
        return BPETokenizer.load(path)
    from .tokenizer import CharTokenizer
    return CharTokenizer.load(path)
