# llm package — Shimba architecture decoder-only transformer
from .compat import enable_safe_output

# Applied at import so library-level calls (GPT.load, generate, Trainer) are
# safe on Windows consoles too, not just the CLI entry points.
enable_safe_output()

from .model import GPT, GPTConfig
from .tokenizer import CharTokenizer
from .data import TextDataset, DataLoader, make_splits
from .train import Trainer, TrainConfig, resume_from
from .generate import generate, stream_generate
from .checkpoint import save_checkpoint, load_checkpoint, unwrap
from .device import resolve_device, describe_device
from .compat import enable_safe_output

__all__ = [
    "GPT", "GPTConfig",
    "CharTokenizer",
    "TextDataset", "DataLoader", "make_splits",
    "Trainer", "TrainConfig", "resume_from",
    "generate", "stream_generate",
    "save_checkpoint", "load_checkpoint", "unwrap",
    "resolve_device", "describe_device",
    "enable_safe_output",
]
