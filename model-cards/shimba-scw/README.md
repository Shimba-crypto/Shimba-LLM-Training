---
license: mit
pipeline_tag: text-generation
language:
  - en
tags:
  - text-generation
  - pytorch
  - tiny
  - from-scratch
---

![shimba-scw](banner.jpg)

# shimba-scw

The same 10.7M Shimba model shipped as **`.scw` (Shimba Container
Weights)** — config, tokenizer and weights in **one file**, no pickle:

```bash
python scw.py pack --model model.pth --out model.scw
python scw.py pack --model model.pth --out model_q8.scw --dtype q8_0
python scw.py info --model model.scw
```

Why `.scw`: one sequential, memory-mapped load instead of unpickling
(measured `0.18s` vs `0.21s` for 10.7M params on CPU); f32 weights share
RAM with the file instead of duplicating it; `q8_0` quarters the file
with blocks decoded in small chunks so RAM stays flat.

## Use

```bash
python chat.py --model model.scw --temperature 0.7
python setup.py generate --model model.scw --prompt "Once upon a time"
```

`slm-chat` (upcoming CLI) loads `.scw` natively: `GPT.load("model.scw")`
plus `load_tokenizer_for("model.scw")` is the whole integration — byte
layout is specified in `llm/scw.py`.

Unpack back to classic files any time:

```bash
python scw.py unpack --model model.scw --out model.pth
```

## Limitations

Same model, same limits as the `.pth` release: short-form English in its
training style, no real reasoning at this size.
