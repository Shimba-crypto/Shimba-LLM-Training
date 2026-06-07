"""
tokenizer.py — Character-level tokenizer (zero external dependencies).

Design:
  * Builds vocabulary from the training corpus (all unique UTF-8 characters).
  * encode() converts a string → list[int]
  * decode() converts list[int] → str
  * Serialises to / from a plain JSON file for portability.

Why char-level?
  * No dependency on `tokenizers` / SentencePiece.
  * Deterministic and inspectable.
  * Perfectly adequate for corpora up to ~50 MB; vocab typically 100-300 tokens.

If you later want BPE, swap this class out — the interface (encode/decode/vocab_size)
stays the same.
"""

import json
import os
from typing import List


class CharTokenizer:
    """
    Character-level tokenizer.

    Special tokens
    --------------
    <PAD> = 0   (padding / ignore index in cross-entropy)
    <UNK> = 1   (unknown character — rare but keeps encoding total)
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self):
        # Will be populated by build() or load()
        self.char2idx: dict = {}
        self.idx2char: dict = {}
        self.vocab_size: int = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self, text: str) -> "CharTokenizer":
        """
        Build vocabulary from a raw text string.
        Always inserts PAD (0) and UNK (1) first, then all chars in sorted order.
        """
        chars = sorted(set(text))
        # Reserve indices 0 and 1 for special tokens
        vocab = [self.PAD_TOKEN, self.UNK_TOKEN] + chars
        self.char2idx = {ch: i for i, ch in enumerate(vocab)}
        self.idx2char = {i: ch for i, ch in enumerate(vocab)}
        self.vocab_size = len(vocab)
        print(f"[tokenizer] vocabulary size: {self.vocab_size} "
              f"(2 special + {len(chars)} chars)")
        return self

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Convert string to list of integer token ids."""
        unk = self.char2idx[self.UNK_TOKEN]
        return [self.char2idx.get(ch, unk) for ch in text]

    def decode(self, ids: List[int]) -> str:
        """Convert list of integer token ids to string."""
        parts = []
        for i in ids:
            ch = self.idx2char.get(i, self.UNK_TOKEN)
            # Skip special tokens in decoded output
            if ch not in (self.PAD_TOKEN, self.UNK_TOKEN):
                parts.append(ch)
        return "".join(parts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise tokenizer vocab to a JSON file."""
        data = {
            "char2idx": self.char2idx,
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[tokenizer] saved → {path}")

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """Load a previously saved tokenizer from JSON."""
        tok = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok.char2idx   = data["char2idx"]
        tok.idx2char   = {int(k): v for k, v in
                          {str(v): k for k, v in data["char2idx"].items()}.items()}
        # Rebuild idx2char properly
        tok.idx2char   = {v: k for k, v in tok.char2idx.items()}
        tok.vocab_size = data["vocab_size"]
        print(f"[tokenizer] loaded ← {path}  (vocab_size={tok.vocab_size})")
        return tok

    # Convenience: save/load tokenizer alongside model checkpoint
    @staticmethod
    def default_path(model_path: str) -> str:
        """Derive tokenizer JSON path from model .pth path."""
        base, _ = os.path.splitext(model_path)
        return base + "_tokenizer.json"
