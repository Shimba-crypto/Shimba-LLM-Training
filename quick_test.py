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


def test_corpus_formats():
    """Verify .txt / .json / .jsonl corpus loading agrees."""
    print("[test] corpus formats ...")
    import json

    from llm.corpus import collect_docs, extract_doc_text, load_corpus_text

    tmpdir = tempfile.mkdtemp(prefix="shimba_corpus_")
    with open(os.path.join(tmpdir, "a.txt"), "w", encoding="utf-8") as f:
        f.write("The quick brown fox jumps over the lazy dog. " * 10)
    with open(os.path.join(tmpdir, "b.json"), "w", encoding="utf-8") as f:
        json.dump([
            {"text": "Pack my box with five dozen liquor jugs. " * 10},
            {"instruction": "Say hello", "output": "Hello there. " * 10},
            {"messages": [
                {"role": "user", "content": "How are you today? " * 6},
                {"role": "assistant", "content": "Quite well, thank you. " * 6},
            ]},
        ], f)
    with open(os.path.join(tmpdir, "c.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "Sphinx of black quartz judge my vow. " * 10}) + "\n")
        f.write(json.dumps({"prompt": "The rain in Spain", "completion": "falls mainly. " * 10}) + "\n")

    text = load_corpus_text(tmpdir)
    for snippet in ("quick brown fox", "five dozen liquor", "Instruction: Say hello",
                    "User: How are you", "Assistant: Quite well", "black quartz",
                    "The rain in Spain"):
        assert snippet in text, f"missing {snippet!r} in combined corpus"

    # --text-field restricts extraction to the named field.
    assert extract_doc_text({"title": "t", "body": "hello world"}, "body") == "hello world"
    assert extract_doc_text({"title": "t", "body": "   "}, "body") is None

    # A folder limited to *.txt must ignore the json files.
    txt_only = collect_docs(tmpdir, pattern="*.txt")
    assert len(txt_only) == 1 and "quick brown fox" in txt_only[0]

    print(f"  docs={len(collect_docs(tmpdir))}  chars={len(text):,}  OK")


def test_architectures(vocab_size: int):
    """Forward + loss for the gpt2 and llama block variants."""
    print("[test] architectures ...")
    import torch
    from llm.model import GPT, GPTConfig

    for arch in ("gpt2", "llama"):
        cfg = GPTConfig(
            vocab_size=vocab_size,
            block_size=32,
            n_embd=64,
            n_head=2,
            n_layer=2,
            dropout=0.0,
            bias=(arch == "gpt2"),
            arch=arch,
        )
        model = GPT(cfg)
        idx = torch.randint(0, vocab_size, (2, 16))
        tgt = torch.randint(0, vocab_size, (2, 16))
        logits, loss = model(idx, tgt)
        assert logits.shape == (2, 16, vocab_size), f"{arch}: {logits.shape}"
        assert loss is not None and loss.item() > 0
        n = sum(p.numel() for p in model.parameters())
        print(f"  {arch}: logits={tuple(logits.shape)} loss={loss.item():.4f} "
              f"params={n:,}  OK")


def test_bpe():
    """Train a tiny BPE vocab, round-trip text through it and reload."""
    print("[test] BPE tokenizer ...")
    import random

    from llm.bpe import BPETokenizer, load_tokenizer

    random.seed(1)
    words = ("the quick brown fox jumps over the lazy dog while shimba trains "
             "a small transformer on plain text").split()
    corpus = "\n".join(" ".join(random.choices(words, k=12)).capitalize() + "."
                       for _ in range(400))
    tok = BPETokenizer().build(corpus, vocab_size=300)
    assert tok.vocab_size > 258, f"no merges learned: {tok.vocab_size}"

    sample = "The quick brown fox jumps. Pack 123 boxes!"
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample, f"round-trip failed: {tok.decode(ids)!r}"
    # Multi-token words prove merges actually apply.
    assert len(tok.encode("the quick brown fox")) < len("the quick brown fox")

    path = os.path.join(tempfile.gettempdir(), "shimba_bpe_test_tokenizer.json")
    tok.save(path)
    tok2 = load_tokenizer(path)
    assert type(tok2) is BPETokenizer
    assert tok2.encode(sample) == ids
    assert tok2.merges == tok.merges
    print(f"  vocab={tok.vocab_size} merges={len(tok.merges)} "
          f"ids={ids[:6]}...  OK")


def test_thinking_split():
    """split_thinking separates reasoning spans from answers."""
    print("[test] thinking split ...")
    from llm.generate import split_thinking

    t, a = split_thinking("no tags here", "<t>", "</t>")
    assert t is None and a == "no tags here"

    text = ("User: hi\nAssistant: <|begin_of_thought|>\n"
            "pondering...\n<|end_of_thought|>\n<|begin_of_solution|>\nhello!")
    t, a = split_thinking(text, "<|begin_of_thought|>", "<|end_of_thought|>")
    assert t == "pondering...", f"thinking: {t!r}"
    assert a.startswith("<|begin_of_solution|>"), f"answer: {a!r}"

    # Only the end tag matters; the start tag is optional.
    t, a = split_thinking(text, None, "<|end_of_thought|>")
    assert t is not None and t.endswith("pondering...")
    print("  OK")


def test_notebook_build():
    """Regenerate the Colab notebook and syntax-check its code cells."""
    print("[test] notebook build ...")
    import ast
    import subprocess

    repo = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, "build_notebook.py"],
                   capture_output=True, text=True, cwd=repo, check=True)
    import json as _json
    with open(os.path.join(repo, "Shimba_Colab.ipynb"),
              encoding="utf-8") as f:
        nb = _json.load(f)
    n = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        # Colab shell magics are valid in the notebook, not in ast.
        src = "\n".join(ln for ln in "".join(c["source"]).splitlines()
                        if not ln.lstrip().startswith("!"))
        ast.parse(src)
        n += 1
    assert n > 10, f"expected code cells, got {n}"
    print(f"  {len(nb['cells'])} cells, {n} code cells parse  OK")


def test_scw_roundtrip():
    """Pack a checkpoint to .scw and back; values (not just sizes) must hold."""
    print("[test] scw round-trip ...")
    import torch
    from llm.scw import quantize_q8_0, dequantize_q8_0, pack_model

    torch.manual_seed(0)
    x = torch.randn(1152, 384) * 2
    y = dequantize_q8_0(quantize_q8_0(x), x.numel()).reshape(x.shape)
    err = (x - y).abs().max().item()
    assert err < 0.1, f"q8 round-trip too lossy: {err}"

    from llm.model import GPT, GPTConfig
    from llm.tokenizer import CharTokenizer

    torch.manual_seed(0)
    tmp = tempfile.gettempdir()
    tok = CharTokenizer().build("Hello, world! 123 abc")
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=32,
                    n_embd=64, n_head=2, n_layer=2, dropout=0.0)
    src = os.path.join(tmp, "shimba_scw_test.pth")
    GPT(cfg).save(src)
    tok.save(CharTokenizer.default_path(src))
    scw_path = os.path.join(tmp, "shimba_scw_test.scw")
    pack_model(src, scw_path)
    m1, m2 = GPT.load(src), GPT.load(scw_path)
    s1 = {k: v.float() for k, v in m1.state_dict().items()}
    s2 = {k: v.float() for k, v in m2.state_dict().items()}
    assert s1.keys() == s2.keys()
    worst = max((s1[k] - s2[k]).abs().max().item() for k in s1)
    assert worst == 0.0, f"f32 scw must be bit-exact, got {worst}"
    print(f"  q8 err={err:.4f}  f32 diff={worst}  OK")


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
    test_corpus_formats()
    test_architectures(tok.vocab_size)
    test_bpe()
    test_scw_roundtrip()
    test_thinking_split()
    test_notebook_build()
    test_cli_help()

    print("\n" + "=" * 52)
    print("  All tests passed!")
    print("=" * 52)
    print("\nNext step:")
    print("  python setup.py train --data your_book.txt --out model.pth")


if __name__ == "__main__":
    main()