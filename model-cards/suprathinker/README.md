---
license: mit
pipeline_tag: text-generation
language:
  - en
tags:
  - text-generation
  - pytorch
  - tiny
  - reasoning
  - from-scratch
---

![suprathinker](banner.jpg)

# suprathinker

A small **thinking** language model: it reasons inside
`<|begin_of_thought|>...<|end_of_thought|>` tags before answering inside
`<|begin_of_solution|>...<|end_of_solution|>`. Trained from scratch with
[Shimba-LLM-Training](https://github.com/Shimba-crypto/Shimba-LLM-Training)
on the [SupraThink](https://huggingface.co/datasets/SupraLabs/SupraThink-Dataset-500x)
conversations (500 reasoning traces, Apache-2.0).

## Use

```bash
python chat.py --model suprathinker.pth --temperature 0.7
```

Split the trace from the answer with `split_thinking` (`llm/generate.py`):

```python
from llm.generate import split_thinking
thinking, answer = split_thinking(
    output, "<|begin_of_thought|>", "<|end_of_thought|>")
```

## Ollama (native thinking field)

Export with the GPT-2-compatible layout, then declare the tags in the
template so Ollama splits `thinking` from `content` by itself:

```bash
python pth2gguf.py --model suprathinker.pth --out suprathinker.gguf \
    --outtype q8_0 --compat gpt2
```

```text
FROM ./suprathinker.gguf
PARAMETER num_ctx 256
PARAMETER temperature 0.5
PARAMETER stop "<|end_of_solution|>"
TEMPLATE """{{/* content.split('</think>') */}}{{- range .Messages }}{{- if eq .Role "user" }}User: {{ .Content }}
{{ end }}{{- if eq .Role "assistant" }}Assistant: {{ end }}{{- if .Thinking }}<|begin_of_thought|>{{ .Thinking }}<|end_of_thought|>{{ end }}{{- if eq .Role "assistant" }}{{ .Content }}
{{ end }}{{- end }}Assistant:"""
```

```bash
ollama create suprathinker -f Modelfile
curl http://localhost:11434/api/chat -d '{
  "model": "suprathinker",
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "think": true, "stream": false
}'  # -> message.thinking + message.content
```

## Training

Colab T4, `Suprathinker_Colab.ipynb` in the repo (preconfigured: BPE
tokenizer so long traces fit the context, GPU mixed precision,
checkpointing that survives disconnects).

## Limitations

Small models learn the *shape* of reasoning (tags, step-by-step rhythm)
before real reasoning. Expect fluent-looking traces that wander; scale
(data, size, iters) is what turns the shape into substance.

## Credits

Reasoning data: SupraLabs/SupraThink-Dataset-500x (Apache-2.0).
