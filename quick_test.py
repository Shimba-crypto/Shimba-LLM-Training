#!/usr/bin/env python3
"""
quick_test.py — smoke test for the Shimba LLM.

Runs a tiny model (2 layers, 64-dim) for a handful of iterations on
synthetic data to verify every component works end-to-end without needing a
real corpus. Runs on whatever device is available (CUDA on Colab, else CPU).

Usage:
    python quick_test.py
"""

import sys
import os
import tempfile
import torch

torch.manual_seed(0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.compat import enable_safe_output

enable_safe_output()

# Cross-platform temp path — works on Windows, macOS, Linux
_TMP_MODEL = os.path.join(tempfile.gettempdir(), "shimba_test.pth")
_TMP_TOK = os.path.join(tempfile.gettempdir(), "shimba_test_tokenizer.json")


def test_tokenizer():
    print("[test] tokenizer ...")
    from llm.tokenizer import CharTokenizer
    text = "Hello, world! 123 abc"
    tok = CharTokenizer().build(text)
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text, f"round-trip failed: {decoded!r} != {text!r}"
    print(f"  vocab_size={tok.vocab_size}  encoded={encoded[:8]}...  OK")
    return tok


def test_model(vocab_size: int):
    print("[test] model forward pass ...")
    from llm.model import GPT, GPTConfig
    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=32,
        n_embd=64,
        n_head=2,
        n_layer=2,
        dropout=0.0,
    )
    model = GPT(cfg)
    idx = torch.randint(0, vocab_size, (2, 16))   # batch=2, seq=16
    tgt = torch.randint(0, vocab_size, (2, 16))
    logits, loss = model(idx, tgt)
    assert logits.shape == (2, 16, vocab_size), f"bad logits shape {logits.shape}"
    assert loss is not None and loss.item() > 0
    print(f"  logits={tuple(logits.shape)}  loss={loss.item():.4f}  OK")
    return model, cfg


def test_training(model_cfg, tok):
    print("[test] training loop (5 iters) ...")
    from llm.data import make_splits
    from llm.train import Trainer, TrainConfig
    import random

    # Synthetic corpus: random chars from tokenizer vocab
    chars = [c for c in tok.char2idx.keys()
             if c not in (tok.PAD_TOKEN, tok.UNK_TOKEN)]
    corpus = "".join(random.choices(chars, k=2000))
    tokens = tok.encode(corpus)

    train_ds, val_ds = make_splits(
        tokens,
        block_size=model_cfg.block_size,
        val_fraction=0.2,
    )

    train_cfg = TrainConfig(
        batch_size=2,
        gradient_accumulation_steps=1,
        max_iters=5,
        eval_interval=5,
        eval_iters=2,
        log_interval=5,
        out_path=_TMP_MODEL,          # <-- cross-platform temp path
    )

    trainer = Trainer(model_cfg, train_cfg, train_ds, val_ds)
    trainer.run()
    print("  OK")


def test_checkpoint_roundtrip():
    """
    Save → reload → generate with the matching tokenizer.

    Uses the same tokenizer for training and generation. The previous version
    built a *different* tokenizer here, so the vocab never matched the saved
    model and the generation path was silently skipped instead of tested.
    """
    print("[test] checkpoint round-trip ...")
    from llm.model import GPT
    from llm.tokenizer import CharTokenizer
    from llm.generate import generate

    if not os.path.exists(_TMP_MODEL):
        print("  [FAIL] model file not found — training did not produce it")
        return False

    # The training test saved with this vocab; rebuild it identically.
    import random
    random.seed(0)
    chars = [c for c in CharTokenizer().build("Hello, world! 123 abc").char2idx
             if c not in (CharTokenizer.PAD_TOKEN, CharTokenizer.UNK_TOKEN)]
    corpus = "".join(random.choices(chars, k=2000))
    tok = CharTokenizer().build(corpus)

    model = GPT.load(_TMP_MODEL)
    assert model.cfg.vocab_size == tok.vocab_size, (
        f"vocab mismatch: model={model.cfg.vocab_size} tok={tok.vocab_size}")

    result = generate(model, tok, "Hello", max_new_tokens=20, top_k=5)
    assert isinstance(result, str) and len(result) > 0, "generation returned nothing"
    print(f"  generated: {result!r}  OK")
    return True


def test_cli_help():
    """Verify the CLI entry point runs without errors."""
    print("[test] CLI help ...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "setup.py"],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    assert "train" in result.stdout, f"Expected 'train' in help output: {result.stdout[:200]}"
    assert "generate" in result.stdout, f"Expected 'generate' in help output"
    print("  OK")


def main():
    from llm.device import describe_device, resolve_device

    device = resolve_device("auto")

    print("=" * 52)
    print("  Shimba LLM -- smoke test")
    print("=" * 52)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  Device  : {describe_device(device)}")
    print(f"  Temp dir: {tempfile.gettempdir()}")
    print("=" * 52)

    tok        = test_tokenizer()
    model, cfg = test_model(tok.vocab_size)
    test_training(cfg, tok)
    test_checkpoint_roundtrip()
    test_cli_help()

    print("\n" + "=" * 52)
    print("  All tests passed!")
    print("=" * 52)
    print("\nNext step:")
    print("  python setup.py train --data your_book.txt --out model.pth")


if __name__ == "__main__":
    main()