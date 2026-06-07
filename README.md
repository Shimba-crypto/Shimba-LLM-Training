# Shimba LLM

A small but complete **decoder-only Transformer** language model that runs entirely on CPU with ≤ 6 GB RAM.

## Architecture

```
Token + Position Embeddings
         │
    ┌────┴────┐
    │  Block  │ × N  (pre-norm: LN → Attention → residual
    │         │                 LN → MLP       → residual)
    └────┬────┘
    LayerNorm
    LM Head  (weight-tied to token embedding)
```

Default hyperparameters:

| Parameter    | Value  | Notes                          |
|--------------|--------|--------------------------------|
| `vocab_size` | auto   | derived from training corpus   |
| `block_size` | 512    | context window (tokens)        |
| `n_embd`     | 256    | embedding / hidden dimension   |
| `n_head`     | 4      | attention heads                |
| `n_layer`    | 4      | transformer blocks             |
| `dropout`    | 0.1    |                                |
| `bias`       | False  | faster, fewer parameters       |

Model size at defaults: **~10 M parameters, ~38 MB float32**

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `torch >= 2.0`, `tqdm` (for progress bars).

---

## Usage

### Train

```bash
python setup.py train \
    --data  corpus.txt \
    --out   model.pth
```

Common overrides:

```bash
# Smaller/faster model for quick experiments
python setup.py train \
    --data corpus.txt \
    --n_embd 128 --n_layer 2 --n_head 2 \
    --max_iters 1000

# Larger model if you have enough RAM
python setup.py train \
    --data corpus.txt \
    --n_embd 512 --n_layer 6 --n_head 8 \
    --block_size 256 --batch_size 4
```

The best checkpoint (lowest val loss) is saved to `--out`.
A final checkpoint is saved to `<stem>_final.pth`.
The tokenizer is saved alongside as `<stem>_tokenizer.json`.

### Generate

```bash
python setup.py generate \
    --model  model.pth \
    --prompt "Once upon a time" \
    --max_tokens 300

# Stream tokens as they are produced
python setup.py generate \
    --model  model.pth \
    --prompt "The quick brown" \
    --stream

# More creative (higher temperature)
python setup.py generate \
    --model model.pth \
    --prompt "Dear reader," \
    --temperature 1.2 --top_k 50 --top_p 0.9
```

---

## Memory estimates

| Config               | Parameters | RAM (float32) |
|----------------------|-----------|----------------|
| Default (256, 4, 4)  | ~10 M      | ~38 MB model  |
| Medium  (512, 6, 8)  | ~85 M      | ~324 MB model |
| Large   (768, 12, 12)| ~85 M      | ~750 MB model |

Plus activations during training (~2–4× model size), tokenized corpus, and PyTorch overhead — all comfortably under 6 GB for the default config even on a 300 MB text file.

---

## Project structure

```
llm_project/
├── setup.py          # CLI entry point
├── requirements.txt
└── llm/
    ├── __init__.py
    ├── model.py      # GPT, CausalSelfAttention, MLP, TransformerBlock
    ├── tokenizer.py  # CharTokenizer (char-level BPE-free)
    ├── data.py       # TextDataset, DataLoader, make_splits
    ├── train.py      # Trainer, TrainConfig, LR schedule
    └── generate.py   # generate(), stream_generate()
```

---

## CPU optimisations

1. **Gradient accumulation** — effective batch 32 without 32× peak memory.
2. **`torch.compile`** — graph-compiled forward/backward on PyTorch ≥ 2.0.
3. **`int16` token storage** — halves corpus RAM vs `int32`.
4. **Weight tying** — LM head shares embedding weights (saves `vocab × embd × 4` bytes).
5. **Cosine LR + warmup** — stable convergence without manual tuning.
6. **Gradient clipping** — prevents loss spikes without restarting.
7. **Best-checkpoint tracking** — saves only when val loss improves.
