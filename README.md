# imagejev

Typed decisions about images, in one forward pass, with calibrated confidence.

You give it an image and a few questions; it answers each as a **choice** (pick one option),
a **score** (pick a level on an ordered scale) or a **yes/no**, and returns probabilities, not text.
It is the image counterpart of [Laya](https://github.com/NandhaKishorM/laya) (text only) and of
TypeSafe's Jev (hosted, no image input yet).

> **Status: research code with an honest negative result.** The library, the benchmark and the
> training pipeline work and are tested. The one trained model so far beats the SigLIP 2 zero-shot
> baseline on questions like the ones it trained on, but **not on task families it has never seen**,
> which is the point of the project. By the rule written into the spec, no weights are published.
> Read [docs/results-v0.md](docs/results-v0.md) before relying on any of it.
>
> "imagejev" is a **working name**; it must change before any release because "Jev" is TypeSafe's
> product name.

## What you can run today

```bash
pip install -e ".[model]"        # from a checkout; the package is not on PyPI
```

```python
from imagejev import Model

m = Model.load("siglip2-zeroshot")      # the zero-shot baseline; also loads exported model directories

result = m.predict(
    image="receipt.jpg",                # path, PIL image, bytes, or a handle from m.encode()
    state={"note": "customer returned the item"},     # optional text or JSON context
    questions={
        "doc_type": {
            "type": "choice",
            "instructions": "What kind of document is this?",
            "criteria": {"receipt": "proof of purchase", "invoice": "bill requesting payment",
                         "other": "anything else"},
        },
        "legibility": {"type": "score", "instructions": "How readable is the text?",
                       "levels": ["unreadable", "poor", "ok", "clear"]},   # ordered low -> high
        "signed": {"type": "bool", "instructions": "Is there a handwritten signature?"},
    },
)
result["doc_type"]   # {"answer": "receipt", "probs": {...}, "confidence": 0.91}
result["legibility"] # {"answer": "ok", "probs": {...}, "expected": 2.3, "confidence": 0.74}
result["signed"]     # {"answer": False, "p_true": 0.12, "confidence": 0.88}
```

**Encode once, ask many.** `h = m.encode("receipt.jpg")` runs the vision encoder once; pass `h` to
`predict` for further questions and only the cheap decision step runs again.

**Escalate when unsure.** `confidence` is the calibrated probability of the returned answer, so the
usual pattern is: answer locally when it is high, and send the rest to a large model.

Limits: one image per call, up to 32 options per question, `state` text is cut at 256 tokens.

## What is in the repository

| | |
|---|---|
| `src/imagejev/` | the API, the SigLIP 2 baseline, the trained model, calibration |
| `src/imagejev/data/` | adapters and generators that turn COCO, Rico, RVL-CDIP and synthetic pages, documents and quality-degraded images into typed questions with **exact** labels; the question templater; the split builder (held-out task families and styles) |
| `src/imagejev/bench/` | metrics, the benchmark runner, baselines (SigLIP 2, SmolVLM), the release-criteria gate |
| `src/imagejev/train/` | balanced sampler, resumable training loop, Kaggle helpers |
| `results/` | benchmark results, with the exact commands to reproduce them |
| `docs/` | the design spec, data licenses and notes, the results write-up |

## How the trained model works

A frozen SigLIP 2 turns the image into 65 cached feature tokens. ModernBERT (with LoRA adapters)
reads the question and each option. Four fusion blocks let each option attend to the image, and one
option-mixing layer (no positional information, so option order cannot matter for choice and yes/no)
produces a logit per option. SigLIP's own zero-shot logit is added as a **prior** and the network
learns a correction on top (it started at the baseline's behaviour; see the spec for why that was
needed). Everything is calibrated after training with a temperature per question type.

## Reproduce

[results/README.md](results/README.md) has the full commands: build the benchmark data, cache the
image features, train, export, benchmark, and run the release-criteria gate.

## Data and licensing

The code is Apache-2.0 ([LICENSE](LICENSE)). **The data it was built from is not**: COCO images are
mostly NonCommercial, Rico screenshots may not be republished, RVL-CDIP documents come from a
library that puts copyright compliance on the user. Nothing here redistributes third-party images.
See [docs/data-licenses.md](docs/data-licenses.md), including the decisions that are still open.

## Known gaps

No Kaggle run yet; no teacher-labelled slice yet; no ablations; one training run and one seed; a
single accuracy bar still awaiting a decision. [docs/results-v0.md](docs/results-v0.md) lists them
with the next steps.
