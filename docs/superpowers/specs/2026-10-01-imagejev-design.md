# imagejev — design spec

Date: 2026-10-01
Status: draft for review
Working name: `imagejev` (must be renamed before public release — "Jev" is TypeSafe's product name)

## 1. Goal

An open-source **System 1 decision engine for images**: typed `choice`, `score` and `bool` decisions over an image (plus optional text state) in a single forward pass, with calibrated probabilities and no text generation.

It is the image counterpart of [Laya](https://github.com/NandhaKishorM/laya) (text-only) and of TypeSafe's Jev, which has no image input.

### What the user decided

- **Purpose:** an open-source release (model weights + `pip` package + benchmark), in the style of Laya.
- **Scope:** general purpose. Photos, documents and screenshots are all first-class, and none of them dominates.
- **Ambition:** a trained model, not just a wrapper around an existing one.
- **Budget:** $0 for v0. All work runs on a MacBook Pro with 24 GB RAM and on free Kaggle/Colab GPU tiers. Spending more is reconsidered only after stage-2 results.
- **Architecture:** approach A, a trained fusion head on a frozen SigLIP 2 (section 3).
- **Teacher:** keep a small slice of examples labelled by a local VLM (section 4.4).

### Success

Section 7.4 defines the release criteria. In short, v0 must beat SigLIP 2 zero-shot on calibration and log loss for unseen tasks in every domain, while running at least 10× faster than a small VLM.

### Non-goals (v0)

- More than one image per call.
- Region or bounding-box answers.
- More than 32 options per question. Jev handles 255; a two-stage approach is deferred.
- Generating free text.
- Batch CLI, FastAPI server and ONNX export. These come after the core works, mirroring Laya.

## 2. Public API

The question schema deliberately matches Laya's, so the same questions work on text or images and the two could later share a router.

```python
from imagejev import Model

m = Model.load("imagejev-base")

result = m.predict(
    image="receipt.jpg",                  # path | PIL.Image | bytes | ImageHandle
    state={"merchant_note": "returned"},  # optional str or JSON-serialisable dict
    questions={
        "doc_type": {
            "type": "choice",
            "instructions": "What kind of document is this?",
            "criteria": {"receipt": "proof of purchase",
                         "invoice": "bill requesting payment",
                         "other": "anything else"},
        },
        "legibility": {
            "type": "score",
            "instructions": "How readable is the text?",
            "levels": ["unreadable", "poor", "ok", "clear"],   # ordered low → high
        },
        "signed": {"type": "bool", "instructions": "Is there a handwritten signature?"},
    },
)
```

Output, keyed by question id:

| type | fields |
|---|---|
| choice | `answer: str`, `probs: dict[str, float]`, `confidence: float` |
| score | `answer: str`, `probs: dict[str, float]`, `expected: float` (0-based level index), `confidence: float` |
| bool | `answer: bool`, `p_true: float`, `confidence: float` |

- `confidence` is the calibrated probability of the returned answer. For `bool` that is `max(p_true, 1 - p_true)`.
- **Encode once, ask many.** `h = m.encode(image)` returns an `ImageHandle` holding the cached vision features. Passing `predict(image=h, ...)` skips the vision encoder. Results must be identical to passing the raw image.
- The README documents the escalation pattern: answer locally when `confidence ≥ threshold`, otherwise call a large VLM.

### Validation and errors

| Condition | Behaviour |
|---|---|
| Image can't be decoded | raise `ImageLoadError` |
| `choice` with no `criteria`, or `score` with fewer than 2 `levels` | raise `QuestionSchemaError` |
| More than 32 options or levels | raise `QuestionSchemaError` |
| Unknown `type` | raise `QuestionSchemaError` |
| `state` longer than the token budget (256 tokens) | truncate it and emit a `UserWarning` |

## 3. Model

```
image ─► SigLIP 2 base/16 @384 (frozen) ─► 24×24 = 576 patch tokens
      ─► 3×3 avg-pool ─► 64 tokens + 1 global token          (cached, fp16, ~100 KB/image)

state ─► text encoder ─► state tokens
memory = [image tokens ; state tokens]

for each option o:
    [instructions ‖ "label: description"] ─► text encoder (ModernBERT-base + LoRA)
    ─► fusion block × 4 (self-attn → cross-attn to memory → MLP), d = 512
    ─► pooled option vector

option vectors ─► 1 option-mixing self-attention layer (no positional encoding)
               ─► linear ─► one logit per option
```

- **Vision:** `SigLIP 2 base patch16-384` (~86M parameters), frozen. Pooling is fixed, not learned, so features can be cached before training.
- **Text:** `ModernBERT-base` (149M parameters) with LoRA adapters (rank 16, on the attention projections). The state text is encoded **once per call**, not once per option.
- **Fusion:** about 25–30M trainable parameters.
- **Permutation invariance:** options are encoded independently with shared weights, and the mixing layer has no position information. Option order therefore cannot change the output. This avoids Laya's position-bias problem by construction.
  **Exception: `score`.** Score levels are ordered by definition (low to high), and a set-based model
  cannot know which end is "high" unless the words say so (`["A", "B", "C"]` would be unorderable).
  Score options therefore also receive a rank embedding (their position divided by K - 1).
  Order-independence is guaranteed and tested for `choice` and `bool`; for `score` the model is
  equivariant when each level keeps its rank, and reversing the ranks does change the answer.
  Release criterion 4 (shuffle test) applies to `choice` and `bool` only.
- **bool:** encoded as a single "option" containing only the instructions. Its logit goes through a sigmoid.
- **Zero-shot prior (added after the pilot).** ModernBERT's features and SigLIP's image features were
  never aligned, and a 1,000-step pilot on bench v1 showed the fusion model does not learn that
  alignment in useful time: `bool` loss stayed at ln 2 for 500 steps (SigLIP's own text tower, which
  *is* aligned with its vision tower, scores AUROC 0.69 on the same questions). The model therefore
  receives SigLIP's zero-shot logit for every option as an input (exactly the baseline's raw logits:
  `"<instructions> <option>"` per option, and instructions minus a generic caption for `bool`) and
  learns a *correction* on top: `logit = prior_gain[type] * prior + head(...)`, with each head's last
  layer zero-initialised. A fresh model therefore makes the same decision as the zero-shot baseline
  on every question (checked on 800 real validation questions) and training can only move it from
  there. Compositional, negated and criteria-following questions, where the zero-shot score is wrong,
  are what the correction learns. This is also what makes release criterion 1 reachable: the model
  starts at the baseline instead of having to beat it from scratch.
- **Heads and losses** (all proper scoring rules):
  - `choice`: softmax over options, log loss.
  - `bool`: sigmoid, binary log loss.
  - `score`: softmax over levels, log loss + ranked probability score (RPS, weight 1.0). `expected` = Σ i·p_i.
- **Calibration:** after training, one temperature per question type is fitted on the validation split and shipped with the weights.
- **Caches:** image features (`ImageHandle`) and question/option encodings, keyed by a hash of the text, are both reusable.
- **Total size (measured):** the fusion model has 23.7M trainable parameters (two 2-layer adapters
  2.6M, four fusion blocks 16.8M, mixer 3.2M, heads and rank embedding 1.1M); with the 1.6M LoRA
  weights that is 25.3M. The earlier estimate of "about 30M" was a rough guess.
- **Latency targets** (to be measured, not yet measured): GPU p50 under 50 ms for one cold image with one question, and under 10 ms for each extra question on an encoded image. Mac numbers are reported too.

## 4. Data

Principle: use labels that are known to be true wherever possible. The teacher labels only what nothing else can.

### 4.1 Existing datasets converted to typed questions

| Domain | Source | Question types |
|---|---|---|
| Photos | COCO annotations | bool (object present), choice (scene / supercategory), score (count bins: none / one / few / many) |
| Photos | VQAv2, yes/no subset | bool, with real human-written questions |
| Screenshots | Rico | choice (screen type), bool (element present) |
| Documents | RVL-CDIP | choice (16 document types) |

### 4.2 Synthetic data with known labels

- **Web screenshots:** generated HTML rendered with Playwright, with controlled state: modal, error banner, spinner, login form, empty cart, captcha box, CTA position above or below the fold.
- **Documents:** templated receipts, invoices and forms. Merchant type, total, signature present, table present and stamp are all known.
- **Image-quality scores:** blur, JPEG compression, noise, darkening and cropping applied at known strengths to any source image.

### 4.3 Question generation

- About 20 paraphrased instructions per task, generated once and stored in version control.
- Random option subsets, varied option descriptions, and an optional "none of these" option.
- Hard distractors drawn from neighbouring classes.
- Bool questions balanced to about 50/50.
- **Compositional questions with exact answers**, built from known labels:
  - logic over COCO labels ("a dog and no person");
  - count thresholds;
  - synthetic document fields ("a restaurant receipt over $50");
  - screenshot layout ("button above the fold").

### 4.4 Teacher-labelled slice

- **Target:** 20–30k examples of judgement-style or open-world criteria on real images.
- **Teacher:** a 7B Qwen-VL model, 4-bit, run through MLX on the Mac.
- **Labels:** the teacher is prompted to answer with one option token. Its probabilities over the option tokens are normalised and stored as soft labels.
- **Quality check:** the user hand-checks about 200 examples. Any task family where the teacher's accuracy on that check is below 80% is dropped.

### 4.5 Splits

| Split | Purpose |
|---|---|
| train | — |
| val | model selection, temperature fitting |
| test-images | unseen images, seen task families |
| val-tasks | five validation task families (disjoint from the test families), used only for model selection |
| test-tasks | whole task families never seen in training: the main criteria-following test |
| test-styles | synthetic templates and themes never seen in training (catches memorising the generator) |
| test-external | three public datasets, zero-shot, never trained on (below) |

Splits are made by source image ID, so no image appears in two splits.

**External test sets (chosen in #22).** Picked for: a license that allows evaluation (checked from
the Hub metadata, recorded in `docs/data-licenses.md`); labels that convert into typed questions
with exact answers; real (not synthetic) images; and a size that downloads in minutes.

| Domain | Dataset | License | Typed questions |
|---|---|---|---|
| Photos | Oxford-IIIT Pet (`timm/oxford-iiit-pet`, test: 3,669) | CC BY-SA 4.0 | species, breed (4- and 8-way, same-species distractors), "is this a *breed*?" |
| Documents | CORD-v2 receipts (`naver-clova-ix/cord-v2`, test + validation: 200) | CC BY 4.0 | item count, tax line, service charge, paid in cash |
| Screenshots | ScreenSpot (`rootsautomation/ScreenSpot`, 610 distinct screenshots) | Apache 2.0 | platform: iOS, Android, macOS, Windows, web |

Checked against the real data: ScreenSpot's `data_source` has eight values that map onto the five
platforms with no conflicts (199 web, 132 Windows, 115 iOS, 86 Android, 78 macOS). CORD's receipts
are Indonesian and priced in rupiah, so total-amount questions are **not** used.
Ruled out: RICO-ScreenQA (inherits Rico's terms), `lmms-lab/VizWiz-VQA` and
`tanganke/stanford_cars` (no license metadata), Food-101 and CIFAR-100 (license `unknown`),
DocLayNet (license `other`), WebSight (synthetic, like our own generator).

### 4.6 Size and licensing

- About 150–200k images and 1–2M question instances. The feature cache is about 20 GB.
- Code and weights will be Apache 2.0. **Every source's license must be checked before release.** Any published dataset includes only the permissively licensed and synthetic parts. ImageNet is excluded.

## 5. Training

1. **Cache features once** on the Mac (MPS). The cache can be rebuilt by rerunning the step.
2. **Sampler:** pick images first, then K = 4 questions per image, sharing the loaded features.
   - Domains (photos / documents / screenshots) are weighted equally.
   - Question types (choice / score / bool) are weighted equally.
   - Teacher examples make up about 15% of each batch.
3. **Optimiser:** AdamW. Learning rate 3e-4 for the fusion layers and 1e-4 for LoRA, with cosine decay and warmup.
4. **Hardware:**
   - Mac: development and small runs in fp32.
   - Kaggle: full runs in fp16, saving a checkpoint every N steps and resuming after the session limit.
5. **Monitoring:** log loss, accuracy and ECE each epoch, broken down by domain × question type.
6. **Model selection:** the lowest macro log loss on `val-tasks`: task families that are held out of training like the test families, but **disjoint from them**, so choosing a checkpoint never touches the test families (see §4.5).
7. **Post-hoc:** fit the per-type temperatures.
8. **Reproducibility:** YAML configs, fixed seeds and CSV logs. No paid tracking services.

### Ablations (short runs on the cached features)

1. 64 vs 144 image tokens.
2. Square input vs SigLIP 2 NaFlex, which keeps the aspect ratio (needs a second cache).
3. With vs without the teacher slice.
4. With vs without the option-mixing layer, plus the option-shuffle test.

## 6. Delivery stages and spending gates

| Stage | Cost | Output | Gate to continue |
|---|---|---|---|
| 0 | $0 | Typed-API wrapper over SigLIP 2 zero-shot (the baseline) | — |
| 1 | $0 | Data pipeline, splits, feature cache, benchmark harness | Baselines run end to end |
| 2 | $0 | Trained fusion model v0 | Release criteria 1–2 (§7.4) on val |
| 3 | $0 (budget decision revisited here) | Ablations, teacher slice, final run, release | All release criteria on test |

## 7. Benchmark

### 7.1 Baselines

1. SigLIP 2 zero-shot behind the same typed API, with its own fitted temperature (stage 0).
2. A small VLM (SmolVLM or moondream), prompted, with option probabilities read from its output.
3. The teacher (7B Qwen-VL), as the accuracy ceiling.

Jev isn't benchmarked because it has no image input.

### 7.2 Metrics

- Accuracy, plus within-one-level accuracy for `score`.
- Log loss and ECE (15 bins).
- Accuracy at 80% coverage, answering only the most confident 80%.
- Latency p50 and p95, cold and warm (encoded image), on the Mac and on a Kaggle T4.

All metrics are reported per domain and per question type.

### 7.3 Splits used

test-images, test-tasks, test-styles, test-external (§4.5).

### 7.4 Release criteria (fixed before training)

1. On **test-tasks**, lower log loss and ECE than SigLIP 2 zero-shot, in **every** domain.
2. No domain falls below SigLIP 2 zero-shot accuracy on any test split.
3. Accuracy within 5 points of the small-VLM baseline at ≥10× lower warm latency. (The 5-point bar may be revised before training starts, never after.)

   > **Proposed revision, pending owner sign-off (#24).** Not yet in force: the original wording
   > above still stands until someone confirms. It must be settled before the first release-run
   > training starts (#35).
   >
   > The baselines are in (`results/`). On `test-images`, accuracy by domain was:
   >
   > | | SigLIP 2 zero-shot | SmolVLM-500M | better of the two |
   > |---|---|---|---|
   > | photo | 0.596 | 0.660 | 0.660 |
   > | document | 0.619 | 0.575 | 0.619 |
   > | screenshot | 0.672 | 0.514 | 0.672 |
   > | all | 0.623 | 0.594 | 0.623 |
   >
   > Warm p50 latency: SigLIP 0.4 ms, SmolVLM 69 ms.
   >
   > "Within 5 points of the small VLM" assumed the VLM was the stronger baseline. It isn't: it is
   > weaker than SigLIP overall (0.594 vs 0.623), so the bar would let a model pass at about 0.55,
   > below SigLIP on two of three domains. **Recommended replacement:** on `test-images`, accuracy in
   > **every domain is at least the better of the two baselines' accuracy in that domain**
   > (recomputed on the final benchmark data; today photo >= 0.660, document >= 0.619,
   > screenshot >= 0.672), **and** warm p50 latency is at most one tenth of the small VLM's
   > (<= 6.9 ms on the same machine). A gentler alternative is "no more than 2 points below the
   > better baseline in any domain". Either is far stricter than the original, which is the point:
   > the original bar was calibrated before there was any data.
4. Shuffling the option order changes the answer in fewer than 1% of cases (`choice` and `bool` questions; `score` levels are ordered, see §3).

**If criterion 1 fails:** v0 is not released as a model. We publish the benchmark and the negative result, then move to approach B: a small VLM used as an encoder that reads answer logits without decoding.

## 8. Testing

- Schema validation and every error path in §2.
- Tensor shapes through every stage.
- Option-shuffle invariance: the same probabilities, up to 1e-5, under permuted option order.
- `predict(encode(img))` gives the same result as `predict(img)`.
- Temperature fitting reduces ECE on a synthetic, deliberately miscalibrated fixture.
- Golden tests: a small set of images with expected answers, as a regression check.

## 9. Repository layout (proposed)

```
imagejev/
  src/imagejev/        # api.py, schema.py, model/, cache.py, calibrate.py
  data/                # builders per source, synthetic generators, question templates
  train/               # configs/, train.py, sampler.py
  bench/               # baselines/, run.py, metrics.py
  tests/
  docs/superpowers/specs/
```

## 10. Open items

- Final project name (must not use "Jev").
- License check for each data source (§4.6).
- Confirm or replace the accuracy bar in §7.4 criterion 3 (proposal written, awaiting sign-off, #24).
