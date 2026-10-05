# Shimba LLM

A small, complete **decoder-only Transformer** language model written from
scratch in PyTorch. Trains on a single GPU (Colab T4 is plenty), on Apple
Silicon, or on a plain CPU.

Tokenisation is character-level by default — no vocabulary file to download
and no tokenizer library to install — with an optional byte-level BPE
tokenizer (`--tokenizer bpe`) for longer effective context.

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

### Architectures (`--arch`)

The block design is selectable at train time; merging requires equal `--arch`:

| Arch | Norm | Positions | Attention | MLP | Bias | Notes |
|------|------|-----------|-----------|-----|------|-------|
| `shimba` (default) | LayerNorm | learned | MHA, fused QKV | GELU 4× | no | the original design |
| `gpt2` | LayerNorm | learned | MHA, fused QKV | GELU 4× | yes | matches HF GPT-2 block-for-block |
| `llama` | RMSNorm | RoPE | MHA/GQA, fused QKV | SwiGLU 4× | no | `n_head_kv` enables grouped-query attention |
| `flash` | RMSNorm | RoPE | GQA by default, fused QKV | SwiGLU 4× | no | the fast arch: 1 kv head per 4 query heads + KV-cache decode path (`flash.py`) |

```bash
python setup.py train --data corpus.txt --out gpt2.pth --arch gpt2
python quick_train.py --data data.jsonl --out llama.pth --arch llama
```

## Presets

| Preset  | `n_embd` | `n_layer` | `n_head` | Params | Use for                     |
|---------|----------|-----------|----------|--------|-----------------------------|
| `tiny`  | 64       | 2         | 2        | ~0.4 M | smoke tests, quick checks   |
| `small` | 256      | 4         | 4        | ~10 M  | the default                 |
| `medium`| 512      | 6         | 8        | ~85 M  | longer context, big corpora |

## Usage

### Train

`--data` accepts a single file **or a folder** of files, which are merged
in sorted order with a separator between documents. Three formats are
supported, detected from the file extension:

- `.txt` — raw text, used as-is
- `.json` — a JSON array of documents, or a single JSON object
- `.jsonl` — one JSON value per line (lines that are not valid JSON are
  kept as plain text)

```bash
python setup.py train --data corpus.txt --out model.pth

# A whole folder of documents
python setup.py train --data ./contracts --out model.pth

# Byte-level BPE tokenizer instead of char-level
python setup.py train --data corpus.txt --out model.pth \
    --tokenizer bpe --vocab-size 2000

# JSONL training data
python setup.py train --data data.jsonl --out model.pth

# Colab GPU with mixed precision
python setup.py train --data corpus.txt --out model.pth --amp

# Resume after an interruption (Colab disconnects without warning)
python setup.py train --data corpus.txt --out model.pth --resume
```

Each JSON document is usually an object with a `text` (or `content`)
field. Objects with `instruction`/`input`/`output`, `prompt`/`completion`,
`question`/`answer`, or `messages`/`conversation` lists are rendered into
plain text automatically. Use `--text-field` to read a specific field
instead (comma-separated names, tried in order), and `--format` to force
one parser for files whose extension does not match their content.

Useful flags:

| Flag | Meaning |
|---|---|
| `--device` | `auto` (default), `cpu`, `cuda`, `cuda:1`, `mps` |
| `--amp` | mixed precision on CUDA — near-free speedup on a T4 |
| `--compile` | `torch.compile`; slower first run, off by default |
| `--resume` | continue from the checkpoint at `--out` |
| `--pattern` | glob(s) for `--data` folders (default `*.txt,*.json,*.jsonl`) |
| `--tokenizer` | `char` (default) or byte-level `bpe` |
| `--vocab-size` | BPE vocabulary size target (default: 2000) |
| `--arch` | `shimba` (default), `gpt2`, `llama` — see Architectures |
| `--text_field` | JSON field(s) to train on (default: auto-detect) |
| `--format` | `auto` (default), `txt`, `json`, `jsonl` |
| `--no_recurse` | only top-level files in a `--data` folder |
| `--max_iters`, `--batch_size`, `--grad_accum`, `--lr` | standard knobs |

Three files get written:

- `<out>` — best checkpoint (lowest validation loss)
- `<stem>_final.pth` — checkpoint at the end of the run
- `<stem>_tokenizer.json` — **required** to load the model

### Tokenizers

`char` (default) maps each character to one id — simple and dependency-free.
`bpe` learns GPT-2-style byte-level merges from your corpus, so common words
become single tokens and the same `block_size` covers more text (about 4–6×
fewer tokens per character on English). The pre-tokenizer pattern
approximates GPT-2's with stdlib `re` only; BPE files store the merge list,
and GGUF export writes it out, so Ollama tokenizes prompts identically
(verified token-for-token against `encode()`).

Resuming requires the matching `--tokenizer`; the saved kind is checked
before training continues.

### Thinking traces

Datasets like
[SupraThink](https://huggingface.co/datasets/SupraLabs/SupraThink-Dataset-500x)
wrap reasoning in tags
(`<|begin_of_thought|>...<|end_of_thought|><|begin_of_solution|>...`).
They train like any other JSONL — the tags are plain text to the model:

```bash
# download (Apache-2.0) and train
python setup.py train --data suprathink.jsonl --out think.pth --tokenizer bpe
```

`split_thinking(text, think_start, think_end)` (in `llm/generate.py`,
demoed in the Colab notebook) separates the reasoning span from the final
answer for display. It returns `(None, text)` when the tags are absent,
so it is safe to call on any output.

### Thinking in Ollama

Ollama splits a separate `thinking` field when the model's template
declares it. For custom tags (SupraThink's `<|begin_of_thought|>` /
`<|end_of_thought|>`), reference `{{ .Thinking }}` wrapped in those tags
inside the `{{ range .Messages }}` loop — Ollama infers the delimiters
from the template:

```text
TEMPLATE """{{/* content.split('</think>') */}}{{- range .Messages }}{{- if eq .Role "user" }}User: {{ .Content }}
{{ end }}{{- if eq .Role "assistant" }}Assistant: {{ end }}{{- if .Thinking }}<|begin_of_thought|>{{ .Thinking }}<|end_of_thought|>{{ end }}{{- if eq .Role "assistant" }}{{ .Content }}
{{ end }}{{- end }}Assistant:"""
```

```bash
ollama create thinkmodel -f Modelfile   # needs PARAMETER stop "<|end_of_solution|>"
curl http://localhost:11434/api/chat -d '{
  "model": "thinkmodel",
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "think": true, "stream": false
}'  # -> message.thinking + message.content
```

Check `/api/show` for `thinking` under `capabilities` — if present, the
template engaged. Two caveats, both verified: the model only splits when
it actually emits both tags (a 6K-iter tiny demo emits the opener
reliably; the closer needs a bigger run — train that on Colab), and
`think: true` requests parsing while `think: false` returns raw text.

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

Tokens print as they are generated by default; pass `--no-stream` to print
the full reply at once instead.

A character-level model has no chat format built in — this is a thin prompt
wrapper, so expect rambling. `--temperature_ramp` gradually raises
temperature over a long conversation, which helps it stop looping.

### Merge

```bash
# equal average of two checkpoints (same architecture required)
python merge.py --models a.pth b.pth --out merged.pth

# weighted average of three
python merge.py --models a.pth b.pth c.pth --weights 0.5 0.3 0.2 --out merged.pth

# slerp between two checkpoints
python merge.py --models a.pth b.pth --method slerp --t 0.5 --out merged.pth
```

Inputs must share `--arch`, `vocab_size`, `block_size`, `n_embd`, `n_head`,
`n_layer`, and their tokenizer JSON files must agree. The tokenizer from
`--tokenizer-from` (default: model 0) is copied alongside the output.
Also available as `python setup.py merge ...`.

### `.scw` — single-file models

`.scw` (Shimba Container Weights) ships the **full** model in one file:
config, tokenizer (char or BPE) and weights, with no pickle. It loads
memory-mapped, so startup is one sequential pass and f32 weights share RAM
with the file instead of duplicating it:

```bash
python scw.py pack --model model.pth --out model.scw
python scw.py pack --model model.pth --out model_q8.scw --dtype q8_0
python scw.py info --model model.scw
python scw.py unpack --model model.scw --out model.pth  # back to .pth + JSON
```

`--dtype f16` halves the file; `q8_0` quarters it (norms stay f32, and
q8 blocks decode in small chunks so RAM stays flat). A `.scw` loads
anywhere a `.pth` does — `chat.py`, `setup.py generate`, and the
`slm-chat` CLI later: `GPT.load("model.scw")` plus
`load_tokenizer_for("model.scw")` is the whole integration. Byte layout
is documented in `llm/scw.py`.

### GGUF export
```bash
python pth2gguf.py --model model.pth --out model.gguf --outtype f32
python pth2gguf.py --model model.pth --out model.gguf --outtype f16
python pth2gguf.py --model model.pth --out model.gguf --outtype q8_0
```

Writes a GGUFv3 file containing the weights and the char-level vocabulary
in one file. `--compat` selects the declared architecture (`auto` follows
the checkpoint: `shimba` stays custom, `gpt2`/`llama` export natively).
`f16` stores 2-D weights as float16, `q8_0` block-quantizes 2-D weights;
norms and biases stay float32 in both modes. Also available as
`python setup.py gguf ...`.

### Ollama

`gpt2` and `llama` checkpoints export to GGUFs that Ollama loads directly.
Char-vocab files embed the vocabulary with an empty BPE merge list, so
prompts tokenize character by character for ASCII text; BPE checkpoints
export their real merge list:

```bash
python pth2gguf.py --model model.pth --out model.gguf --outtype q8_0  # --compat auto
```

```text
# Modelfile (keep num_ctx within the model's block_size)
FROM ./model.gguf
PARAMETER num_ctx 256
```

```bash
ollama create mymodel -f Modelfile
ollama run mymodel "Once upon a time"
```

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
├── setup.py              # CLI: train / generate / merge / gguf / help
├── quick_train.py        # checkpoint-every-step trainer (Colab-friendly)
├── quick_test.py         # smoke test, no corpus required
├── chat.py               # interactive chat
├── flash.py              # pack / info / generate for *_flash.pth fast models
├── quantize.py           # int8 / fp16 export
├── merge.py              # average / slerp checkpoint merge
├── scw.py                # .scw pack / info / unpack
├── pth2gguf.py           # GGUF export (f32 / f16 / q8_0)
├── build_notebook.py     # regenerates Shimba_Colab.ipynb
├── Shimba_Colab.ipynb    # Colab notebook
├── model-cards/          # HuggingFace-ready cards + banners
│   ├── shimba-10M/       # .pth release card
│   └── shimba-scw/       # .scw single-file card
├── requirements.txt
└── llm/
    ├── __init__.py
    ├── model.py          # GPT, CausalSelfAttention, MLP, TransformerBlock
    ├── tokenizer.py      # CharTokenizer (char-level, zero dependencies)
    ├── bpe.py            # byte-level BPE training + BPETokenizer
    ├── data.py           # TextDataset, DataLoader, make_splits
    ├── corpus.py         # .txt / .json / .jsonl corpus loading
    ├── train.py          # Trainer, TrainConfig, resume_from
    ├── generate.py       # generate(), iter_generate(), stream_generate()
    ├── flash.py          # flash arch helpers, KVCache, fast_generate(), pack/load
    ├── checkpoint.py     # canonical save/load, legacy-format tolerance  
    ├── scw.py            # .scw single-file format (mmap, q8_0)
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