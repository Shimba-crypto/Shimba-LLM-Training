"""
data.py — Dataset utilities for the Shimba LLM.

TextDataset
-----------
  Stores the entire corpus as a flat int16 tensor (saves ~50% RAM vs int32/int64
  when vocab_size ≤ 32767, which is true for char-level tokenisers).

  Each call to __getitem__ returns a (context, target) pair where:
    context = tokens[i : i + block_size]          (int64 for embedding lookup)
    target  = tokens[i + 1 : i + block_size + 1]  (int64, shifted by 1)

DataLoader
----------
  A lightweight, dependency-free data loader that:
    * Randomly samples starting positions each epoch (no fixed ordering).
    * Returns CPU tensors only.
    * Supports train/val splits by index range.

Memory budget example (8 GB corpus, char-level):
  int16 tensor ≈ corpus_chars × 2 bytes ≈ 16 MB for 8 M characters.
  For a 100 MB text file that's ~200 MB — well within our 6 GB budget.
"""

import random
import torch
from typing import Tuple


class TextDataset:
    """
    Flat token sequence stored as an int16 tensor.

    Parameters
    ----------
    tokens : list[int]
        All token ids for the corpus (train or val split).
    block_size : int
        Number of tokens per training context window.
    """

    def __init__(self, tokens: list, block_size: int):
        # int16 halves RAM vs int64. Guard the boundary: a very large corpus can
        # push vocab_size past 32767 (heavy multilingual text), and silently
        # wrapping to negative ids would corrupt every embedding lookup.
        if tokens and max(tokens) > 32767:
            print("[data] vocab exceeds int16 range — storing as int32")
            dtype = torch.int32
        else:
            dtype = torch.int16
        # We cast to int64 on __getitem__ so embedding layers work.
        self.data       = torch.tensor(tokens, dtype=dtype)
        self.block_size = block_size

    def __len__(self) -> int:
        # Number of valid starting positions
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1].to(torch.int64)
        x = chunk[:-1]   # context
        y = chunk[1:]    # target (one-ahead)
        return x, y


def make_splits(
    tokens: list,
    block_size: int,
    val_fraction: float = 0.1,
) -> Tuple[TextDataset, TextDataset]:
    """
    Split a flat token list into train and validation TextDatasets.

    Parameters
    ----------
    tokens : list[int]   – full encoded corpus
    block_size : int     – context window length
    val_fraction : float – fraction of data to use for validation

    Returns
    -------
    train_dataset, val_dataset
    """
    n = len(tokens)
    split = int(n * (1 - val_fraction))
    train_tokens = tokens[:split]
    val_tokens   = tokens[split:]

    train_ds = TextDataset(train_tokens, block_size)
    val_ds   = TextDataset(val_tokens,   block_size)

    print(f"[data] train tokens: {len(train_tokens):,}  "
          f"val tokens: {len(val_tokens):,}")
    print(f"[data] train batches available: {len(train_ds):,}  "
          f"val batches available: {len(val_ds):,}")
    return train_ds, val_ds


class DataLoader:
    """
    Minimal random-sampling data loader — no multiprocessing (CPU only).

    Each call to __iter__ yields `batch_size` random (x, y) pairs
    packed into (B, T) tensors.

    Using random sampling (rather than sequential) means we never have
    to shuffle a large list — we just draw random start indices.
    """

    def __init__(self, dataset: TextDataset, batch_size: int, shuffle: bool = True):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.shuffle    = shuffle

    def __len__(self) -> int:
        return max(1, len(self.dataset) // self.batch_size)

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a single random batch.

        Returns
        -------
        x : (batch_size, block_size) int64
        y : (batch_size, block_size) int64
        """
        n = len(self.dataset)
        if n == 0:
            raise ValueError("Dataset is empty — check your text file.")
        idxs = [random.randint(0, n - 1) for _ in range(self.batch_size)]
        xs, ys = zip(*[self.dataset[i] for i in idxs])
        return torch.stack(xs), torch.stack(ys)

    def __iter__(self):
        """Iterate over one 'epoch' (len(self) batches)."""
        for _ in range(len(self)):
            yield self.get_batch()


def load_text(path: str) -> str:
    """
    Read a text file with graceful encoding fallback.
    Tries UTF-8 first, then latin-1 (which never fails).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        print("[data] UTF-8 decode failed — retrying with latin-1")
        with open(path, "r", encoding="latin-1") as f:
            text = f.read()

    print(f"[data] loaded '{path}'  ({len(text):,} characters)")
    return text
