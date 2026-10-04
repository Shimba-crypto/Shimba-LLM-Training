#!/usr/bin/env python3
"""
build_notebook.py — generates Shimba_Colab.ipynb.

Kept as a generator so the notebook JSON is always valid and the cells stay
readable/reviewable as Python strings.
"""

import json

cells = []


def _as_source(text):
    """Split into notebook source lines.

    Each entry must keep its trailing newline (except the last) or every line
    collapses into one when the cell is read back.
    """
    lines = text.strip("\n").splitlines()
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _as_source(text)})


def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _as_source(text),
    })


md(r"""
# Shimba LLM — Google Colab

Train a decoder-only transformer from scratch on any text, on Colab's GPU.

**Before you start:** pick `Runtime > Change runtime type > T4 GPU`. Everything
still runs on CPU without it, just much slower.

How it works:

1. Clone the repo and install PyTorch
2. Point it at a text corpus (upload, Google Drive, or a URL)
3. Train — checkpoints save every iteration, so a disconnect is recoverable
4. Generate, chat, and download the model

> Colab resets the filesystem when a runtime is deleted. Mount Drive (step 4)
> and your corpus plus checkpoints will still be there next time.
""")

code(r'''
# ── Setup ────────────────────────────────────────────────────────────
import os, sys, subprocess, shutil

REPO = "https://github.com/Shimba-crypto/Shimba-LLM-Training.git"
WORKDIR = "/content/shimba-llm"

if os.path.isdir(os.path.join(WORKDIR, "llm")):
    print("Repo already present — pulling latest changes")
    subprocess.run(["git", "-C", WORKDIR, "pull", "--ff-only"], check=False)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, WORKDIR], check=True)

os.chdir(WORKDIR)
sys.path.insert(0, WORKDIR)
print("Working directory:", os.getcwd())
''')

code(r'''
# Colab ships CUDA-matched PyTorch already. Installing with --index-url picks up
# the wheel that matches the runtime's CUDA, which is faster than the default.
!pip install -q torch --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || \
  !pip install -q "torch>=2.0.0"
!pip install -q -r requirements.txt

import torch
print("PyTorch:", torch.__version__)
''')

md(r"""
## 1. Confirm the GPU

Colab hands you a T4 by default. You should see `cuda: 0`.
""")

code(r'''
import torch
from llm.device import resolve_device, describe_device

DEVICE = resolve_device("auto")
print("Device:", describe_device(DEVICE))

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM free: {free/1024**3:.1f} GB / {total/1024**3:.1f} GB")
else:
    print("No GPU found — training will be slow. Check Runtime > Change runtime type.")
''')

code(r'''
# Sanity check: tokenizer + model + a 5-iteration training loop.
!python quick_test.py
''')

md(r"""
## 2. Mount Google Drive (recommended)

Your corpus and checkpoints persist here, so a disconnected or deleted
runtime doesn't lose your work.

Skip this if you just want a quick test — but then anything you train dies
when the runtime ends.
""")

code(r'''
from google.colab import drive

drive.mount("/content/drive")
print("Drive mounted")
''')

md(r"""
## 3. Configure

Edit `CORPUS_PATH` to point at your text: a `.txt`, `.json` or `.jsonl`
file, or a **folder** of them which get merged in sorted order.

Not sure what to use? Leave `CORPUS_PATH = None` and the next cell generates
a small sample corpus so you can confirm the whole pipeline works first.
""")

code(r'''
# ── Edit these ───────────────────────────────────────────────────────
REPO_DIR   = "/content/shimba-llm"
DATA_DIR   = "/content/drive/MyDrive/shimba"     # corpus + checkpoints live here
RUN_NAME   = "shimba_run1"                       # change to start a fresh model
CORPUS_PATH = None                               # e.g. DATA_DIR + "/corpus.txt"

# Tokenizer: "char" (default, one id per character) or "bpe" (subwords,
# longer effective context; needs VOCAB_SIZE below).
TOKENIZER = "char"
VOCAB_SIZE = 2000     # BPE vocabulary target (ignored for char)

# Block architecture: "shimba" (default), "gpt2" or "llama".
ARCH = "shimba"

# Model size — bigger = slower, more capacity.
#   tiny   ~0.4M params   fast, good for a first run
#   small ~10M params     the README default
#   medium ~85M params    needs a bigger corpus to be worth it
PRESET = "small"

PRESETS = {
    "tiny":   dict(n_embd=64,  n_layer=2, n_head=2,  block_size=128, batch_size=32, grad_accum=1),
    "small":  dict(n_embd=256, n_layer=4, n_head=4,  block_size=256, batch_size=16, grad_accum=2),
    "medium": dict(n_embd=512, n_layer=6, n_head=8,  block_size=256, batch_size=8,  grad_accum=4),
}

MODEL_PATH = f"{DATA_DIR}/{RUN_NAME}.pth"

CFG = dict(
    **PRESETS[PRESET],
    max_iters=3000,
    lr=3e-4,
    eval_interval=250,
    eval_iters=40,
    amp=True,            # mixed precision — near-free speedup on a T4
)

for k, v in CFG.items():
    print(f"  {k:16s} {v}")
print(f"\n  model out -> {MODEL_PATH}")
''')

code(r'''
os.makedirs(DATA_DIR, exist_ok=True)
print("Data dir ready:", DATA_DIR)
''')

md(r"""
## 4. Get a corpus

Pick **one** of the options below. `.json` and `.jsonl` work wherever
`.txt` does — each JSON document becomes one training document.

**A — Google Drive:** drop a file (or a folder of them) into `shimba/` in
Drive, then set `CORPUS_PATH` above.

**B — Upload:** run the next cell and choose a file from your computer.

**C — URL:** for public domain books, Project Gutenberg is a good source.

**D — HuggingFace:** download a dataset file directly (JSONL included).
""")

code(r'''
# ── B: upload from your computer ─────────────────────────────────────
from google.colab import files

CORPUS_PATH = f"{DATA_DIR}/corpus.txt"
uploaded = files.upload()

if uploaded:
    # Notebook stores bytes keyed by filename; write them to Drive.
    with open(CORPUS_PATH, "wb") as f:
        for name, data in uploaded.items():
            f.write(data)
            print("wrote", name, len(data), "bytes")
    print("CORPUS_PATH =", CORPUS_PATH)
else:
    print("Nothing uploaded — keeping CORPUS_PATH as configured.")
''')

code(r'''
# ── C: download from a URL (optional) ────────────────────────────────
# Example: Moby Dick, ~1.2 MB of text.
# URL = "https://www.gutenberg.org/files/2701/2701-0.txt"

URL = None
if URL:
    import urllib.request
    CORPUS_PATH = f"{DATA_DIR}/corpus.txt"
    urllib.request.urlretrieve(URL, CORPUS_PATH)
    print("downloaded ->", CORPUS_PATH)
''')

code(r'''
# ── D: HuggingFace dataset file (optional) ─────────────────────────
# Example: SupraThink thinking traces (500 rows of conversations with
# <|begin_of_thought|>...<|end_of_thought|> reasoning, JSONL, Apache-2.0).
# HF_URL = "https://huggingface.co/datasets/SupraLabs/SupraThink-Dataset-500x/resolve/main/data.jsonl"

HF_URL = None
if HF_URL:
    import urllib.request
    CORPUS_PATH = f"{DATA_DIR}/" + HF_URL.rsplit("/", 1)[-1]
    urllib.request.urlretrieve(HF_URL, CORPUS_PATH)
    print("downloaded ->", CORPUS_PATH)
''')

code(r'''
# ── A: sample corpus, so you can test the pipeline with zero setup ───
if CORPUS_PATH is None or not os.path.exists(CORPUS_PATH):
    print("No corpus configured — generating a sample so you can test the flow.")
    import random
    random.seed(0)
    words = ("the quick brown fox jumps over the lazy dog while shimba trains a "
             "small transformer on plain text and learns the shape of language "
             "one character at a time until the loss stops moving").split()
    lines = [" ".join(random.choices(words, k=random.randint(6, 16))).capitalize() + "."
             for _ in range(6000)]
    CORPUS_PATH = f"{DATA_DIR}/sample_corpus.txt"
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("sample corpus ->", CORPUS_PATH)

size_mb = os.path.getsize(CORPUS_PATH) / 1024**2
print(f"Corpus: {CORPUS_PATH}  ({size_mb:.2f} MB)")
''')

md(r"""
## 5. Train

`quick_train.py` writes a checkpoint every iteration and records the
optimizer state, so **Ctrl+C is safe** — the next run with `--resume` picks up
at the exact iteration it stopped.

That matters on Colab: free runtimes get killed without warning.

First run? Lower `CFG["max_iters"]` to `300` and confirm the loss is dropping
before committing to a long one.
""")

code(r'''
# Build the command from CFG so the two can't drift apart.
cmd = [
    "python", "-u", "quick_train.py",
    "--data",  CORPUS_PATH,
    "--out",   MODEL_PATH,
    "--tokenizer", TOKENIZER,
    "--arch",  ARCH,
    "--device", "cuda" if DEVICE.type == "cuda" else "cpu",
    "--max_iters",    str(CFG["max_iters"]),
    "--lr",           str(CFG["lr"]),
    "--n_embd",       str(CFG["n_embd"]),
    "--n_layer",      str(CFG["n_layer"]),
    "--n_head",       str(CFG["n_head"]),
    "--block_size",   str(CFG["block_size"]),
    "--batch_size",   str(CFG["batch_size"]),
    "--grad_accum",   str(CFG["grad_accum"]),
    "--eval_interval", str(CFG["eval_interval"]),
    "--eval_iters",   str(CFG["eval_iters"]),
    "--save_every",   "50",
]
if TOKENIZER == "bpe":
    cmd += ["--vocab-size", str(VOCAB_SIZE)]
if CFG["amp"]:
    cmd.append("--amp")
if os.path.exists(MODEL_PATH):
    cmd.append("--resume")

print(" ".join(cmd))
print(f"live log -> {DATA_DIR}/{RUN_NAME}.log  "
      f"(tail it from another cell if this looks stuck)\n")

# Tee: stream to the cell AND to a log file on Drive, so a silent cell
# never hides progress. `tail -30` the log from any other cell.
log_path = f"{DATA_DIR}/{RUN_NAME}.log"
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1)
with open(log_path, "w", encoding="utf-8") as log_f:
    log_f.write(" ".join(cmd) + "\n\n")
    log_f.flush()
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_f.write(line)
rc = proc.wait()
if rc != 0:
    raise subprocess.CalledProcessError(rc, cmd)
print(f"\ndone — full log: {log_path}")
''')

md(r"""
**Interrupted?** Re-run the cell above — `--resume` is added automatically
whenever `MODEL_PATH` already exists.

Loss should trend downward. It won't be smooth: batches are sampled randomly,
so the number bounces. Judge it by the `val=` column at each eval.
""")

md(r"""
## 6. Generate

A model this size produces fluent-looking fragments, not coherent prose.
Lower `temperature` for repetitive-but-plausible text, raise it for noise.
""")

code(r'''
import torch
from llm.model import GPT
from llm.bpe import load_tokenizer
from llm.tokenizer import CharTokenizer
from llm.generate import generate, split_thinking

tok = load_tokenizer(CharTokenizer.default_path(MODEL_PATH))
model = GPT.load(MODEL_PATH, device=DEVICE)

params = sum(p.numel() for p in model.parameters())
print(f"{params:,} parameters | vocab {tok.vocab_size} | block {model.cfg.block_size}\n")

for prompt in ["The", "Once upon a time", "In the beginning"]:
    print(f"--- {prompt!r} ---")
    print(generate(model, tok, prompt, max_new_tokens=300, temperature=0.7))
    print()
''')

md(r"""
### Thinking traces (optional)

Models trained on thinking datasets (e.g. SupraThink) wrap reasoning in
tags. `split_thinking` separates the reasoning span from the final answer.
""")

code(r'''
# Set these to your dataset's tags (SupraThink shown), or leave None.
THINK_START = "<|begin_of_thought|>"
THINK_END = "<|end_of_thought|>"

prompt = "User: Explain photosynthesis in one sentence.\nAssistant:"
out = generate(model, tok, prompt, max_new_tokens=300, temperature=0.7)
thinking, answer = split_thinking(out, THINK_START, THINK_END)
if thinking is not None:
    print("--- thinking ---")
    print(thinking)
    print("--- answer ---")
print(answer)
''')

md(r"""
## 7. Chat

A character-level model has no notion of a chat format — this is just the
`User:/Assistant:` prompt wrapper from `chat.py`, and expect it to wander.
Useful as a demo, not as an assistant.
""")

code(r'''
def chat(prompt, max_tokens=300, temperature=0.7, top_k=40):
    text = generate(model, tok, prompt, max_new_tokens=max_tokens,
                    temperature=temperature, top_k=top_k, top_p=0.95)
    return text[len(prompt):].lstrip()

def converse(user_input, history=None, temperature=0.7):
    history = history if history is not None else []
    prompt = "\n".join(
        [f"User: {u}" if role == "user" else f"Assistant: {a}" for role, u in history]
    ) + f"\nUser: {user_input}\nAssistant:"
    reply = chat(prompt, temperature=temperature)
    history.extend([("user", user_input), ("assistant", reply)])
    return reply, history

# Non-interactive example:
reply, history = converse("Hello, who are you?")
print(reply)
''')

code(r'''
# Interactive loop — comment this cell out to skip it.
history = []
try:
    while True:
        line = input(">>> ").strip()
        if line in ("/exit", "/quit", ""):
            break
        if line == "/reset":
            history = []
            print("[reset]")
            continue
        reply, history = converse(line, history)
        print("\n" + reply + "\n")
except (EOFError, KeyboardInterrupt):
    print("\nbye")
''')

md(r"""
## 8. Merge two runs (optional)

Average two checkpoints into one. Both must share `--arch` and tokenizer —
e.g. a base run plus its continuation, or two seeds of the same corpus.
Skip this section if you only trained once.
""")

code(r'''
# ── Edit these ───────────────────────────────────────────────────────
MERGE_A = MODEL_PATH          # first checkpoint
MERGE_B = None                # e.g. f"{DATA_DIR}/shimba_run2.pth"
MERGED_PATH = f"{DATA_DIR}/{RUN_NAME}_merged.pth"

if MERGE_B and os.path.exists(MERGE_A) and os.path.exists(MERGE_B):
    import subprocess
    subprocess.run([
        "python", "-u", "merge.py",
        "--models", MERGE_A, MERGE_B,
        "--out", MERGED_PATH,
    ], check=True)
    print("merged ->", MERGED_PATH)
else:
    print("Set MERGE_B to a second checkpoint to merge, else skip.")
''')

md(r"""
## 9. Export to GGUF (optional)

Writes a `.gguf` next to the model for Ollama. `--compat gpt2` loads
`shimba`-arch models in Ollama; `llama`-arch models export natively
(`--compat` defaults to following the checkpoint, so usually omit it).
""")

code(r'''
# ── Edit these ───────────────────────────────────────────────────────
EXPORT_GGUF = False
GGUF_MODEL = MERGED_PATH if os.path.exists(f"{DATA_DIR}/{RUN_NAME}_merged.pth") else MODEL_PATH
GGUF_PATH = os.path.splitext(GGUF_MODEL)[0] + ".gguf"
GGUF_OUTTYPE = "q8_0"    # f32 | f16 | q8_0
GGUF_COMPAT = "auto"     # auto | none | gpt2 | llama

if EXPORT_GGUF:
    from pth2gguf import convert
    convert(GGUF_MODEL, GGUF_PATH, outtype=GGUF_OUTTYPE, compat=GGUF_COMPAT)
    print(f"gguf -> {GGUF_PATH} ({os.path.getsize(GGUF_PATH)/1024**2:.2f} MB)")
    print("Download it, then:  ollama create mymodel -f Modelfile")
    print('Modelfile content:  FROM ./<name>.gguf  +  PARAMETER num_ctx <block_size>')
else:
    print("Set EXPORT_GGUF = True to export, else skip.")
''')

md(r"""
## 10. Yarn-art image demo (optional)

The [Norod78/Yarn-art-style](https://huggingface.co/datasets/Norod78/Yarn-art-style)
dataset pairs yarn-art images with captions. This repo trains on **text**,
so the demo below shows the images with their captions (the Loader-ready
text side), and saves the captions as JSONL you can train on.
Image pixels themselves are not trainable here — no vision encoder exists.
""")

code(r'''
# ── Yarn-art gallery + captions JSONL ──────────────────────────────
SHOW_YARN = False
YARN_N = 6   # images to display

if SHOW_YARN:
    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess as _sp
        _sp.run(["pip", "install", "-q", "datasets"], check=True)
        from datasets import load_dataset

    ds = load_dataset("Norod78/Yarn-art-style", split="train")
    print(f"rows: {len(ds)}")

    from IPython.display import display
    caps = []
    for i, row in enumerate(ds):
        if i >= YARN_N:
            break
        print(f"[{i}] {row['text']}")
        display(row["image"])   # PIL image renders inline in Colab
        caps.append({"text": row["text"]})

    import json as _json
    caps_path = f"{DATA_DIR}/yarn_captions.jsonl"
    with open(caps_path, "w", encoding="utf-8") as _f:
        for c in caps:
            _f.write(_json.dumps(c) + "\n")
    print(f"captions -> {caps_path} (train on it with --data pointing there)")
else:
    print("Set SHOW_YARN = True to load and display the dataset, else skip.")
''')

md(r"""
## 11. Save

Two `.pth` files and one tokenizer — **all three are needed**. The tokenizer
is derived from your corpus, so a model without it can't be loaded.

Keep them together in the same folder.
""")

code(r'''
# Copy into Drive so a new runtime can pick them up.
shutil.copy2(MODEL_PATH, MODEL_PATH)                       # already there
tokenizer_path = CharTokenizer.default_path(MODEL_PATH)
for f in (MODEL_PATH, tokenizer_path):
    print(f"{os.path.getsize(f)/1024**2:7.2f} MB  {f}")
''')

code(r'''
# Download to your machine (pick a folder when prompted).
try:
    from google.colab import files
    files.download(MODEL_PATH)
    files.download(CharTokenizer.default_path(MODEL_PATH))
    print("Downloaded — keep both files in the same folder.")
except Exception as e:
    print("Download skipped:", e)
''')

md(r"""
## Next steps

Raise quality by changing one thing at a time:

| Want | Change |
|---|---|
| Better text | More/longer corpus. Loss is data-limited, not size-limited. |
| Finer detail | Larger `block_size` (512+) so it sees more context |
| More capacity | `PRESET = "medium"`, and raise `max_iters` |
| Less overfitting | Raise `dropout`, lower `n_layer` |
| Faster epochs | Lower `block_size`, raise `batch_size` |

To use the model on your own machine afterwards, see the README — the
checkpoint is plain CPU tensors and loads anywhere:

```bash
pip install -r requirements.txt
python setup.py generate --model model.pth --prompt "The"
```

If a run got cut off, set `RUN_NAME` to a new value and retrain from scratch —
or keep the same name and pass `--resume` to continue.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "name": "Shimba_Colab.ipynb",
            "provenance": [],
            "toc_visible": True,
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open("Shimba_Colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
    f.write("\n")

print(f"Wrote Shimba_Colab.ipynb — {len(cells)} cells")