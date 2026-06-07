# llm package — Shimba architecture decoder-only transformer
from .model import GPT, GPTConfig
from .tokenizer import CharTokenizer
from .data import TextDataset, DataLoader
from .train import Trainer, TrainConfig
from .generate import generate

__all__ = [
    "GPT", "GPTConfig",
    "CharTokenizer",
    "TextDataset", "DataLoader",
    "Trainer", "TrainConfig",
    "generate",
]
