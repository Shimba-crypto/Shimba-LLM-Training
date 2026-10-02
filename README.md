# Shimba LLM

A small, complete **decoder-only Transformer** language model written from
scratch in PyTorch. Trains on a single GPU (Colab T4 is plenty), on Apple
Silicon, or on a plain CPU.

The tokenisation is character-level, so there is no vocabulary file to
download and no tokenizer library to install — the model learns from raw
text.

## Quick start (Google Colab)

The fastest path — no local setup:

1. Open `Shimba_Colab.ipynb` in Colab
2. `Runtime > Change runtime type > T4 GPU`
3. Run the cells top to bottom

The notebook clones the repo, optionally mounts Google Drive so your work
survives a runtime reset, lets you upload or download a corpus, trains, and
shows sample output.

> Colab's filesystem is wiped when a runtime is deleted. Mount Drive before
> training, or your corpus and checkpoints are gone.

## Quick start (local)

```bash
git clone https://github.com/Shimba-crypto/Shimba-LLM-Training.git
cd Shimba-LLM-Training
pip install -r requirements.txt

python quick_test.py                      # smoke test, no corpus needed

python setup.py train \
    --data  corpus.txt \
    --out   model.pth

python setup.py generate \
    --model model.pth \
    --prompt "Once upon a time"
```

## Architecture

```
Token + Position Embeddings
         │
    ┌────┴────┐
    │  Block  │  × N  (pre-norm: LN → Attention → residual
    │         │                 LN → MLP       → residual)
    └────┬────┘
    LayerNorm
    LM Head  (weight-tied to token embedding)
```

Defaults (`small` preset):

| Parameter    | Value  | Notes                          |
|--------------|--------|--------------------------------|
| `vocab_size` | auto   | derived from training corpus   |
| `block_size` | 256    | context window (tokens)        |
| `n_embd`     | 256    | embedding / hidden dimension   |
| `n_head`     | 4      | attention heads                |
| `n_layer`    | 4      | transformer blocks             |
| `dropout`    | 0.1    |                                |
| `bias`       | False  | faster, fewer parameters       |

~10 M parameters, ~38 MB in float32.

## Presets

| Preset  | `n_embd` | `n_layer` | `n_head` | Params | Use for                     |
|---------|----------|-----------|----------|--------|-----------------------------|
| `tiny`  | 64       | 2         | 2        | ~0.4 M | smoke tests, quick checks   |
| `small` | 256      | 4         | 4        | ~10 M  | the default                 |
| `medium`| 512      | 6         | 8        | ~85 M  | longer context, big corpora |

## Usage

### Train

`--data` accepts a single `.txt` file **or a folder** of `.txt` files, which
are merged in sorted order with a separator between documents.

```bash
python setup.py train --data corpus.txt --out model.pth

# A whole folder of documents
python setup.py train --data ./contracts --out model.pth

# Colab GPU with mixed precision
python setup.py train --data corpus.txt --out model.pth --amp

# Resume after an interruption (Colab disconnects without warning)
python setup.py train --data corpus.txt --out model.pth --resume
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--device` | `auto` (default), `cpu`, `cuda`, `cuda:1`, `mps` |
| `--amp` | mixed precision on CUDA — near-free speedup on a T4 |
| `--compile` | `torch.compile`; slower first run, off by default |
| `--resume` | continue from the checkpoint at `--out` |
| `--pattern` | glob for `--data` folders (default `*.txt`) |
| `--no_recurse` | only top-level files in a `--data` folder |
| `--max_iters`, `--batch_size`, `--grad_accum`, `--lr` | standard knobs |

Three files get written:

- `<out>` — best checkpoint (lowest validation loss)
- `<stem>_final.pth` — checkpoint at the end of the run
- `<stem>_tokenizer.json` — **required** to load the model

### Checkpointing that survives a disconnect

`quick_train.py` saves after every iteration (throttled to at most one write
per 60s) and includes optimizer state, so an interrupted run continues from
the exact iteration it stopped:

```bash
python quick_train.py --data corpus.txt --out runs/shimba.pth
python quick_train.py --data corpus.txt --out runs/shimba.pth --resume
```

Ctrl+C is safe at any point.

### Generate

```bash
python setup.py generate \
    --model  model.pth \
    --prompt "Once upon a time" \
    --max_tokens 300

# Stream token by token
python setup.py generate --model model.pth --prompt "The" --stream

# Higher temperature = more random
python setup.py generate --model model.pth --prompt "Dear reader," \
    --temperature 1.0 --top_k 40 --top_p 0.9
```

### Chat

```bash
python chat.py --model model.pth --temperature 0.7
```

`/reset` clears the conversation, `/exit` quits.

A character-level model has no chat format built in — this is a thin prompt
wrapper, so expect rambling. `--temperature_ramp` gradually raises
temperature over a long conversation, which helps it stop looping.

### Quantize

```bash
python quantize.py --model model.pth --out model_int8.pth --dtype int8
python quantize.py --model model.pth --out model_fp16.pth --dtype float16
```

An int8 checkpoint only loads back into an int8-quantized model — you cannot
`GPT.load()` it directly. float16 loads normally but only really helps on GPU.

## Project structure

```
.
├── setup.py              # CLI: train / generate / help
├── quick_train.py        # checkpoint-every-step trainer (Colab-friendly)
├── quick_test.py         # smoke test, no corpus required
├── chat.py               # interactive chat
├── quantize.py           # int8 / fp16 export
├── build_notebook.py     # regenerates Shimba_Colab.ipynb
├── Shimba_Colab.ipynb    # Colab notebook
├── requirements.txt
└── llm/
    ├── __init__.py
    ├── model.py          # GPT, CausalSelfAttention, MLP, TransformerBlock
    ├── tokenizer.py      # CharTokenizer (char-level, zero dependencies)
    ├── data.py           # TextDataset, DataLoader, make_splits
    ├── train.py          # Trainer, TrainConfig, resume_from
    ├── generate.py       # generate(), stream_generate()
    ├── checkpoint.py     # canonical save/load, legacy-format tolerance
    ├── device.py         # CUDA / MPS / CPU resolution
    └── compat.py         # Windows console encoding shim
```

## Design notes

**Weight tying.** The LM head shares its weight matrix with the token
embedding, so there is no second `vocab × n_embd` parameter to learn from
scratch.

**Gradient accumulation.** An effective batch of 32 without holding 32
sequences of activations at once.

**Cosine LR with warmup**, gradient clipping at `max_norm=1.0`, and
best-checkpoint tracking on validation loss.

**`int16` token storage.** Halves the RAM held by the tokenized corpus versus
`int32`. Falls back to `int32` automatically if a corpus ever exceeds the
32767 range.

**Portable checkpoints.** Weights are always moved to CPU before being
written, so a model trained on a Colab GPU loads on a laptop.

**Legacy format tolerance.** `checkpoint.py` accepts weights stored under
`state_dict`, `model`, or `weights`, and strips the `_orig_mod.` prefix that
`torch.compile` adds — so old checkpoints and compiled runs both load.

## Expectations

This is a ~10 M parameter character-level model. On a few MB of text it
learns spelling, word shapes, and sentence rhythm, and it will happily loop.

It will not write coherent paragraphs. That is a model-size and data limit,
not a bug — check `val=` in the training log before blaming the code.

## Requirements

- Python 3.10+
- `torch >= 2.0`
- `tqdm`

No GPU required.

## License

MIT — see [LICENSE](LICENSE).