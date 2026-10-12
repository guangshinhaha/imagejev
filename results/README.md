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

## Follow-up: more task families and a shrinkage penalty (pilot v2, v3)

Both were trained and benchmarked on **bench v2** (bench v1 plus the eight fact-based task families
from #86). Their `test-tasks` set is the same 2,495 questions the v1 numbers use, so the rows below
are comparable. `test-tasks` all-domain, calibrated on `val-tasks`:

| model | accuracy | log loss | ECE |
|---|---|---|---|
| SigLIP 2 zero-shot | 0.426 | 1.153 | 0.069 |
| pilot v1 | 0.346 | 1.251 | 0.148 |
| pilot v2 (more families) | 0.354 | 1.285 | 0.184 |
| pilot v3 (+ `correction_l2: 1.0`) | 0.425 | 1.133 | 0.072 |

More task families alone (v2) did not fix the held-out-family regression. Shrinking the learned
correction toward the SigLIP prior (v3) brings it to parity with the baseline: accuracy is level and
log loss is slightly lower. That is not yet a win, so criterion 2 (beat the baseline on unseen
families) is still not met. One seed, one run; the 0.001 accuracy gap and 0.02 log-loss gap are within
noise. **Correction:** a second seed of the same recipe (`configs/abl_l2_1_seed1.yaml`, benchmarked
below) scores 0.398 / 1.183 / 0.135, so the parity was not reproduced on that seed: it is about 3 points of
accuracy below SigLIP (0.426) with ECE 0.135 vs 0.069. The difference comes from which checkpoint the
loop kept (step 250 for the parity run, 875 for this one); see the teacher-slice section below. (The runs use test sets of 2,492 and 2,495 questions.) Full reports: `imagejev-pilot-v2/`, `imagejev-pilot-v3/`. Configs: `configs/pilot_v2.yaml`,
`configs/pilot_v3.yaml`.

### Ablation: teacher slice, and what actually drives the run-to-run variation (#33, #35, partial)

Pilot v3's recipe (1,000 steps, `correction_l2: 1.0`), with and without the Qwen2.5-VL-7B teacher slice
(6,999 questions labelled, 5,187 kept in bench v3 `train`; 5% or 15% of each batch). Eight runs, all
benchmarked on the bench v3 test splits and calibrated on `val-tasks`; accuracy / log loss / ECE.
"Best step" is the checkpoint the training loop kept (lowest `val-tasks` macro log loss, from the
untracked `runs/*/select_log.csv`). Two controls (rows 1 and 4 of the table) were trained on bench v2, so their `train`
set differs from v3's by more than the teacher records; the other six runs are on bench v3.

| run | best step | `test-tasks` (unseen families) | `test-images` |
|---|---|---|---|
| control, seed 0 (pilot v3, bench v2) | 250 | 0.427 / 1.132 / 0.069 | 0.580 / 0.826 / 0.016 |
| control, seed 2 (bench v3) | 375 | 0.442 / 1.145 / 0.061 | 0.592 / 0.822 / 0.021 |
| control, seed 0 (bench v3) | 750 | 0.397 / 1.188 / 0.138 | 0.677 / 0.746 / 0.076 |
| control, seed 1 (bench v2) | 875 | 0.398 / 1.183 / 0.135 | 0.688 / 0.744 / 0.090 |
| control, seed 1 (bench v3) | 1000 | 0.403 / 1.169 / 0.118 | 0.693 / 0.747 / 0.100 |
| 15% teacher, seed 0 | 750 | 0.395 / 1.174 / 0.130 | 0.677 / 0.752 / 0.080 |
| 5% teacher, seed 0 | 875 | 0.391 / 1.197 / 0.146 | 0.681 / 0.745 / 0.084 |
| 15% teacher, seed 1 | 875 | 0.397 / 1.208 / 0.143 | 0.685 / 0.740 / 0.084 |

For reference, SigLIP 2 zero-shot scores 0.426 / 1.153 / 0.069 on `test-tasks` (2,495 questions).

**The runs split into two groups, and the kept checkpoint step lines up with the split.** The two runs
whose best checkpoint came early (step 250, 375) have the highest `test-tasks` accuracy (0.427, 0.442:
0.427 is level with SigLIP's 0.426 and 0.442 is 1.6 points above it, both with SigLIP-like ECE, 0.069
and 0.061 vs 0.069; but each is a single run) and the
lowest `test-images` accuracy (about 0.59, below SigLIP's 0.606 on the v1-sized `test-images`
split, which is not strictly comparable). The six runs whose best checkpoint came at step 750 or later
all sit in a tight cluster at `test-tasks` 0.391-0.403 and `test-images` 0.677-0.693, with ECE higher on
both splits. All three teacher runs fall in the late cluster, as do three of the five controls (the
late controls are `ctl-v3-s0`, step 750; `v3-seed1-on-v3`, step 875, trained on bench v2; and
`ctl-v3-s1`, step 1000). Inside that cluster, the three teacher runs vs the three late controls
differ by about half a point of accuracy (0.394 vs 0.399 mean), 0.013 of log loss (1.193 vs 1.180) and
0.01 of ECE (0.140 vs 0.130). That is small next to the group split, but three runs per side cannot
resolve an effect of a point or less, so a small teacher effect is not ruled out. What the data do
support is that the earlier write-up (#95, #97, #98: "teacher slice hurts unseen families") is
consistent with having compared against one early-checkpoint control, and that **no teacher-slice effect
larger than the run-to-run spread shows up here**.

Caveats: this is an observation across eight runs, not a controlled experiment. The kept step was not
varied on purpose; no teacher run has an early checkpoint, so "teacher with an early step" was never
observed; the `val-tasks` curve is nearly flat across steps (for example 1.072 at step 375 and 1.073 at
875 for one control), so the step it picks is close to arbitrary; and whether a given run does better
on unseen families at an earlier step *within the same run* has not been tested (the trace runs
described below do that). The earlier `correction_l2: 0` pilots (v1, v2) also kept step 250 and were
the worst on `test-tasks` (0.346, 0.354), so "early is better on unseen families" can at most hold
for the `correction_l2: 1.0` recipe. Also: only 5% and 15% teacher shares; teacher labels not
hand-checked (#34); six of the eight runs are on bench v3 and two controls on v2. Reports:
`imagejev-pilot-v3-on-v3/`, `imagejev-pilot-v3-seed1-on-v3/`, `imagejev-pilot-ctl-v3-s0/`, `-s1/`,
`-s2/`, `imagejev-pilot-teacher/`, `imagejev-pilot-teacher-f05/`, `imagejev-pilot-teacher-seed1/`.

**This was tested; see the next subsection.**

#### Unseen-family score over training (fixed-step snapshots, #35)

Three no-teacher controls on bench v3 (`configs/pilot_trace_s{0,1,2}.yaml`, seeds 0-2) were trained
with `snapshot_every: 125`, which keeps a weights-only snapshot every 125 steps, and with `test-tasks`
logged (never used for selection) at every evaluation. Each snapshot was then benchmarked the standard
way (calibrated on `val-tasks`). `test-tasks`, all-domain, accuracy / log loss / ECE:

| step | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| 125 | 0.428 / 1.137 / 0.071 | 0.414 / 1.150 / 0.083 | 0.412 / 1.155 / 0.092 | 0.418 / 1.147 / 0.082 |
| 250 | 0.411 / 1.151 / 0.103 | 0.399 / 1.163 / 0.109 | 0.391 / 1.169 / 0.128 | 0.400 / 1.161 / 0.113 |
| 1000 | 0.396 / 1.190 / 0.140 | 0.393 / 1.179 / 0.137 | 0.391 / 1.197 / 0.152 | 0.393 / 1.189 / 0.143 |
| SigLIP 2 zero-shot | | | | 0.426 / 1.153 / 0.069 |

(The raw, uncalibrated per-evaluation curves in `runs/pilot_trace_s*/eval_log.csv`, not committed, show
the same shape. Reports: `imagejev-trace_s<seed>_<step>/`.)

**Within a run, the unseen-family score gets worse as training goes on, on every metric, in all three
seeds.** From step 125 to step 1000, accuracy falls by 2.1-3.2 points (mean 2.5), log loss rises by
0.03-0.05 and ECE rises 1.7-2 times. This is consistent with the between-run pattern above (runs that
kept an early checkpoint scored higher on `test-tasks`). It also fits the earlier finding that the
learned correction adds nothing on unseen families: the longer it trains, the further it moves the
model from the SigLIP prior in a direction that does not transfer.

**It never beats the baseline.** The best point, step 125, is about level with SigLIP: accuracy 0.418 vs
0.426 (below), log loss 1.147 vs 1.153 (slightly better), ECE 0.082 vs 0.069 (worse). So stopping early
recovers parity at best, not the win that release criteria 1-2 require, and costs seen-family accuracy
(`test-images` accuracy was about 0.59 for the early-kept runs against 0.68-0.69 for the late ones).
Limits: three seeds, one recipe (`correction_l2: 1.0`, no teacher slice); only steps 125, 250 and 1000
were benchmarked; and with the cosine schedule "step 125" is a model that has barely left warm-up,
not a short run. The traces' kept step differs from the earlier control runs with the same seeds
(for example seed 2 kept step 875 here and step 375 before), because changing the evaluation sets
changes the run, so a seed is not one fixed trajectory.

### Ablation: correction shrinkage strength (#35, partial)

Same recipe as pilot v3 (bench v2, 1,000 steps, seed 0), varying only `correction_l2`. Best
checkpoint value of the selection metric, `val-tasks` macro log loss (lower is better). These are
selection-metric numbers from the training logs (`runs/*/select_log.csv`), **not** full benchmark
runs; only l2 = 1.0 was benchmarked in full (above).

| `correction_l2` | best `val-tasks` macro log loss | config |
|---|---|---|
| 0.3 | 1.008 | `configs/abl_l2_0p3.yaml` |
| 1.0 (pilot v3) | 1.059 | `configs/pilot_v3.yaml` |
| 1.0, seed 1 | 1.079 | `configs/abl_l2_1_seed1.yaml` |
| 3.0 | 1.109 | `configs/abl_l2_3.yaml` |
| 10.0 | 1.122 | `configs/abl_l2_10.yaml` |

Seed noise on this metric is at least 0.02 (seed 0 vs seed 1 at l2 = 1.0). Stronger shrinkage (3, 10)
is worse on the selection metric. Weaker shrinkage (0.3) is better on it by about 0.05, **but that
did not transfer**: benchmarked in full (`imagejev-pilot-v4/`), l2 = 0.3 is worse than l2 = 1.0 on
the held-out `test-tasks`:

| model | accuracy | log loss | ECE |
|---|---|---|---|
| SigLIP 2 zero-shot | 0.426 | 1.153 | 0.069 |
| pilot v3 (l2 = 1.0) | 0.425 | 1.133 | 0.072 |
| pilot v4 (l2 = 0.3) | 0.396 | 1.226 | 0.177 |

Screenshots drive it (accuracy 0.335 -> 0.276, ECE 0.064 -> 0.235). So `val-tasks` (1,165 questions,
few families) is too small and too unlike `test-tasks` to tune this knob on: selecting on it picks
a setting that looks better there and is worse on unseen families. l2 = 1.0 remains the best
benchmarked setting, and is only at parity with the baseline. **Correction:** a second seed of
l2 = 1.0 scores `test-tasks` 0.398 / 1.183 / 0.135, the same as l2 = 0.3 (0.396 / 1.226 / 0.177) on
accuracy, so the l2 = 0.3 vs 1.0 difference in this table is not distinguishable from seed variation (log loss
and ECE differ by 0.043 and 0.042 between them, against 0.051 and 0.066 between the two l2 = 1.0
seeds; l2 = 0.3 has one seed, and the selection metric still prefers it). Shrinking at all (v1 -> v3,
0.346 -> 0.398-0.425; v1 is one seed) still looks like it helps, by more than the seed spread.

## Caveats

- Short pilot runs (1,000 steps), one or two seeds per setting. No error bars; differences of a point or two in the
  small domains are not significant.
- `test-tasks` is small for some families (`comp.photo.more_of` has 41 questions).
- No Kaggle T4 latency yet (#20), the teacher slice (#33) and a few ablations (#35) have been run at two or three seeds per setting at most; the
  differences reported are mostly within the run-to-run variation, which is large (see above).
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
