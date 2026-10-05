# llm package — Shimba architecture decoder-only transformer
from .compat import enable_safe_output

# Applied at import so library-level calls (GPT.load, generate, Trainer) are
# safe on Windows consoles too, not just the CLI entry points.
enable_safe_output()

from .model import GPT, GPTConfig, ARCHES
from .tokenizer import CharTokenizer
from .bpe import BPETokenizer, load_tokenizer
from .scw import (
    load_scw,
    load_tokenizer_for,
    pack_model,
    scw_info,
    unpack_model,
)
from .data import TextDataset, DataLoader, make_splits
from .corpus import (
    collect_docs,
    collect_files,
    extract_doc_text,
    load_corpus_text,
    load_file_docs,
    DEFAULT_PATTERN,
    DOC_SEPARATOR,
)
from .train import Trainer, TrainConfig, resume_from
from .sft import (SFTTrainer, SFTDataset, load_sft_pairs, make_sft_splits,
                  render_pair, encode_pair, render_corpus_text)
from .generate import generate, stream_generate, iter_generate
from .flash import (
    fast_generate,
    KVCache,
    pack_flash,
    load_flash,
    read_flash_meta,
    is_flash_file,
    flash_info,
    describe_flash,
    cache_bytes_per_token,
)
from .lora import (
    LoRALinear,
    inject_lora,
    lora_parameters,
    adapter_state,
    load_adapter_state,
    merge_lora_,
    adapter_info,
    DEFAULT_TARGETS as LORA_DEFAULT_TARGETS,
)
from .checkpoint import save_checkpoint, load_checkpoint, unwrap
from .device import resolve_device, describe_device
from .compat import enable_safe_output
from .vida_compute import (find_vda, vda_available, vda_eval, verify_rows)

__all__ = [
    "GPT", "GPTConfig", "ARCHES",
    "CharTokenizer",
    "BPETokenizer", "load_tokenizer",
    "load_scw", "load_tokenizer_for", "pack_model",
    "scw_info", "unpack_model",
    "TextDataset", "DataLoader", "make_splits",
    "collect_docs", "collect_files", "extract_doc_text",
    "load_corpus_text", "load_file_docs",
    "DEFAULT_PATTERN", "DOC_SEPARATOR",
    "Trainer", "TrainConfig", "resume_from",
    "SFTTrainer", "SFTDataset", "load_sft_pairs", "make_sft_splits",
    "render_pair", "encode_pair", "render_corpus_text",
    "generate", "stream_generate", "iter_generate",
    "fast_generate", "KVCache", "pack_flash", "load_flash",
    "read_flash_meta", "is_flash_file", "flash_info",
    "describe_flash", "cache_bytes_per_token",
    "LoRALinear", "inject_lora", "lora_parameters", "adapter_state",
    "load_adapter_state", "merge_lora_", "adapter_info",
    "LORA_DEFAULT_TARGETS",
    "find_vda", "vda_available", "vda_eval", "verify_rows",
    "save_checkpoint", "load_checkpoint", "unwrap",
    "resolve_device", "describe_device",
    "enable_safe_output",
]
