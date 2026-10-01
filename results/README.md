# Benchmark results

All numbers are produced by this repository's own harness (`python -m imagejev.bench.run`) on
**bench v1**, and every table below is generated from the `results.json` files in this folder by
`python -m imagejev.bench.compare`, not typed by hand.

## Headline

The **pilot model** (a 1,000-step training run, best checkpoint chosen on `val-tasks`) is clearly
better than SigLIP 2 zero-shot on questions like the ones it trained on, **not better on task
families it has never seen, and worse on the held-out test families**. By the project's own rule
(spec section 7.4, criterion 1) it is therefore **not a releasable model**, and no weights are
published. The honest summary and what to try next are in `docs/results-v0.md`.

## What is here

| Folder | What |
|---|---|
| `siglip2-zeroshot/` | the zero-shot baseline, full splits |
| `imagejev-pilot-v1/` | the pilot model, full splits (`gate_val.md`, `gate_test-tasks.md` are the gate reports) |
| `subset300/` | all three models (adding SmolVLM-500M) on **identical** subsets of at most 300 images per split |
| `bench_v1_splits_report.json` | sizes of every split and which held-out families are thin |
| `archive-bench-v0/` | superseded first run, kept for provenance only |

## Protocol (the same for every model)

- **Data:** bench v1 = 12,000 generated web pages, 12,000 generated documents, 4,807 COCO val photos,
  3,312 Rico screens, 16,764 quality-degraded copies and 2,010 external images (Oxford Pets, CORD-v2
  receipts, ScreenSpot). Questions per split: train 85,491, val 4,808, val-tasks
  1,165, test-images 6,732, test-tasks 2,495, test-styles
  23,474, test-external 4,698. Splits are by source image; `test-tasks`
  and `val-tasks` hold task families that are never in training (spec section 4.5).
- **Calibration** is fitted on `val-tasks` for every model (a temperature per question type, Platt
  scaling for yes/no). Fitting on seen-task `val` would make a trained model overconfident on exactly
  the unseen criteria the test splits measure.
- **Subsets:** SmolVLM-500M runs at about 2 images/s, so the three-way comparison uses a fixed,
  hash-chosen subset (at most 300 images per split) that is **identical for every model**. The full
  splits are run for the two fast models.
- Everything is **evaluation-only** third-party data; see `docs/data-licenses.md`. COCO val is not
  license-filtered here, which is fine for reporting numbers but not for anything that ships.

## Results

Cells are accuracy / log loss / ECE. Lower is better for the last two.

### Three models on identical subsets

**test-images**

| domain | n | SmolVLM-500M | SigLIP 2 zero-shot | imagejev pilot |
|---|---|---|---|---|
| photo | 219 | 0.584 / 0.874 / 0.126 | 0.543 / 0.945 / 0.099 | 0.676 / 0.800 / 0.136 |
| document | 444 | 0.565 / 0.793 / 0.049 | 0.599 / 0.770 / 0.066 | 0.633 / 0.688 / 0.066 |
| screenshot | 416 | 0.486 / 1.026 / 0.120 | 0.591 / 0.723 / 0.062 | 0.659 / 0.604 / 0.073 |
| all | 1079 | 0.538 / 0.900 / 0.048 | 0.585 / 0.787 / 0.058 | 0.652 / 0.678 / 0.060 |

_cells are accuracy / log loss / ECE_

**test-tasks**

| domain | n | SmolVLM-500M | SigLIP 2 zero-shot | imagejev pilot |
|---|---|---|---|---|
| photo | 90 | 0.344 / 1.235 / 0.151 | 0.556 / 1.220 / 0.144 | 0.556 / 2.051 / 0.196 |
| document | 126 | 0.357 / 1.323 / 0.094 | 0.484 / 1.013 / 0.055 | 0.492 / 0.991 / 0.075 |
| screenshot | 185 | 0.303 / 1.234 / 0.114 | 0.292 / 1.235 / 0.098 | 0.184 / 1.231 / 0.205 |
| all | 401 | 0.329 / 1.262 / 0.062 | 0.411 / 1.162 / 0.071 | 0.364 / 1.340 / 0.141 |

_cells are accuracy / log loss / ECE_

**test-external**

| domain | n | SmolVLM-500M | SigLIP 2 zero-shot | imagejev pilot |
|---|---|---|---|---|
| photo | 522 | 0.531 / 0.985 / 0.161 | 0.923 / 0.250 / 0.094 | 0.939 / 0.252 / 0.118 |
| document | 74 | 0.351 / 1.091 / 0.181 | 0.459 / 1.243 / 0.286 | 0.473 / 1.213 / 0.309 |
| screenshot | 93 | 0.333 / 1.534 / 0.197 | 0.882 / 0.384 / 0.066 | 0.892 / 0.410 / 0.090 |
| all | 689 | 0.485 / 1.071 / 0.165 | 0.868 / 0.375 / 0.077 | 0.882 / 0.377 / 0.091 |

_cells are accuracy / log loss / ECE_

Latency p50 (one MacBook Pro, MPS): SmolVLM-500M: cold 241 ms, warm 109.7 ms; SigLIP 2 zero-shot: cold 69 ms, warm 0.8 ms; imagejev pilot: cold 74 ms, warm 11.0 ms

### SigLIP 2 vs the pilot on the full splits

**val**

| domain | n | SigLIP 2 zero-shot | imagejev pilot |
|---|---|---|---|
| photo | 1101 | 0.596 / 0.828 / 0.042 | 0.715 / 0.674 / 0.075 |
| document | 1988 | 0.580 / 0.766 / 0.054 | 0.620 / 0.669 / 0.033 |
| screenshot | 1719 | 0.614 / 0.712 / 0.027 | 0.690 / 0.583 / 0.050 |
| all | 4808 | 0.596 / 0.761 / 0.033 | 0.667 / 0.639 / 0.032 |

_cells are accuracy / log loss / ECE_

**val-tasks**

| domain | n | SigLIP 2 zero-shot | imagejev pilot |
|---|---|---|---|
| photo | 170 | 0.647 / 0.748 / 0.083 | 0.629 / 0.750 / 0.072 |
| document | 738 | 0.545 / 0.878 / 0.099 | 0.547 / 0.898 / 0.108 |
| screenshot | 257 | 0.693 / 0.734 / 0.154 | 0.685 / 0.740 / 0.167 |
| all | 1165 | 0.592 / 0.828 / 0.040 | 0.590 / 0.842 / 0.050 |

_cells are accuracy / log loss / ECE_

**test-images**

| domain | n | SigLIP 2 zero-shot | imagejev pilot |
|---|---|---|---|
| photo | 1435 | 0.585 / 0.857 / 0.054 | 0.706 / 0.681 / 0.069 |
| document | 2772 | 0.595 / 0.784 / 0.047 | 0.641 / 0.668 / 0.031 |
| screenshot | 2525 | 0.629 / 0.701 / 0.026 | 0.691 / 0.583 / 0.049 |
| all | 6732 | 0.606 / 0.769 / 0.033 | 0.674 / 0.639 / 0.038 |

_cells are accuracy / log loss / ECE_

**test-tasks**

| domain | n | SigLIP 2 zero-shot | imagejev pilot |
|---|---|---|---|
| photo | 506 | 0.559 / 1.180 / 0.138 | 0.601 / 1.593 / 0.153 |
| document | 841 | 0.454 / 1.026 / 0.070 | 0.433 / 1.002 / 0.090 |
| screenshot | 1148 | 0.347 / 1.234 / 0.058 | 0.170 / 1.284 / 0.219 |
| all | 2495 | 0.426 / 1.153 / 0.069 | 0.346 / 1.251 / 0.148 |

_cells are accuracy / log loss / ECE_

**test-styles**

| domain | n | SigLIP 2 zero-shot | imagejev pilot |
|---|---|---|---|
| document | 15370 | 0.531 / 0.894 / 0.068 | 0.577 / 0.756 / 0.031 |
| screenshot | 8104 | 0.626 / 0.702 / 0.019 | 0.702 / 0.576 / 0.056 |
| all | 23474 | 0.564 / 0.828 / 0.048 | 0.620 / 0.694 / 0.029 |

_cells are accuracy / log loss / ECE_

**test-external**

| domain | n | SigLIP 2 zero-shot | imagejev pilot |
|---|---|---|---|
| photo | 3600 | 0.927 / 0.262 / 0.104 | 0.936 / 0.262 / 0.122 |
| document | 488 | 0.463 / 1.140 / 0.233 | 0.469 / 1.104 / 0.211 |
| screenshot | 610 | 0.836 / 0.462 / 0.033 | 0.841 / 0.474 / 0.033 |
| all | 4698 | 0.867 / 0.379 / 0.062 | 0.875 / 0.377 / 0.078 |

_cells are accuracy / log loss / ECE_

Latency p50 (one MacBook Pro, MPS): SigLIP 2 zero-shot: cold 61 ms, warm 0.7 ms; imagejev pilot: cold 74 ms, warm 10.0 ms

## How to read this

- `val`, `test-images`, `test-styles` test **seen task families on unseen images or styles**; the
  pilot wins there in every domain (for example test-images accuracy 0.606 -> 0.674, log loss
  0.769 -> 0.639).
- `val-tasks` and `test-tasks` test **task families never seen in training**; this is the project's
  thesis. The pilot ties on `val-tasks` and loses on `test-tasks` (accuracy 0.426 -> 0.346, ECE
  0.069 -> 0.148; screenshots 0.347 -> 0.170).
- `test-external` is real data nobody trained on. The pilot's edge is small (0.867 -> 0.875) and
  within noise for the smaller domains.
- SmolVLM-500M is the weakest model everywhere here and collapses on external data (0.485).
- **Latency** is from one MacBook Pro under heavy background load, so the absolute numbers are
  pessimistic: an unloaded run measured 5.7 ms warm for a comparable checkpoint.

## Caveats

- One training run, one seed, 1,000 steps. No error bars; differences of a point or two in the
  small domains are not significant.
- `test-tasks` is small for some families (`comp.photo.more_of` has 41 questions).
- No Kaggle T4 latency yet (#20), no teacher-labelled slice yet (#33), no ablations (#35).
- The pilot's checkpoint was selected on `val-tasks`; calibration was also fitted there. The test
  splits were not used for either.

## Reproduce

```bash
python -m imagejev.data.build --out data/bench_v1 --n-web 12000 --n-docs 12000 \
    --coco-dir data/raw/coco --rico-parquet data/raw/rico_vh.parquet --rico-limit 3312 \
    --quality-sources 3000 --external
python -m imagejev.cache --source data/bench_v1/facts_all.jsonl:/ --out data/cache_v1
python -m imagejev.train.loop --config configs/pilot_v1.yaml --fresh
python - <<'PY'
from imagejev.backends.trained import export_model
export_model("runs/pilot_v1/best.pt", "models/pilot")
PY
python -m imagejev.bench.run --model models/pilot --splits-dir data/bench_v1/splits \
    --fit-on val-tasks --save-calibration --images data/bench_v1/facts_all.jsonl:/ --out results/pilot
python -m imagejev.bench.gate --candidate results/pilot/results.json \
    --baseline results/siglip2-zeroshot/results.json --split test-tasks
```
