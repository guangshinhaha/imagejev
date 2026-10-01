# Benchmark results

Each folder is one model run through the same harness (`python -m imagejev.bench.run`).

| Folder | Model |
|---|---|
| `siglip2-zeroshot/` | SigLIP 2 base/16 @384, zero-shot behind the typed API (the baseline) |

`bench_v0_splits_report.json` describes the data these numbers come from (question and image counts
per split, and which held-out task families are still too small to measure; see `thin_families`).

## Reproduce

```bash
# 1. data: ~38k train, 1.9k val and 11k test questions across photos, documents and screenshots
python -m imagejev.data.build --out data/bench_v0 --n-web 3000 --n-docs 3000 \
    --coco-dir data/raw/coco --rico-parquet data/raw/rico_vh.parquet --quality-sources 300
# 2. run a model (temperatures are fitted on the val split)
python -m imagejev.bench.run --model siglip2-zeroshot --splits-dir data/bench_v0/splits \
    --splits val test-images test-tasks test-styles \
    --images data/bench_v0/facts_all.jsonl:/ --out results/siglip2-zeroshot
```

`data/raw/coco` needs `instances_val2017.json` and the `val2017/` images; `rico_vh.parquet` is one
shard of the Hub config `ui-screenshots-and-view-hierarchies`. COCO val2017 is used here for
**evaluation only** and is not license-filtered; anything that ships must follow
`docs/data-licenses.md`. Latencies were measured on one MacBook Pro (MPS) and will differ elsewhere.

## Caveats on bench_v0

- Seven of the nine held-out task families have fewer than 100 test examples, so `test-tasks`
  numbers are noisy. The real build needs several times more synthetic pages and documents.
- No external test set yet (`test-external` is empty; see #22), and no RVL-CDIP documents.
- Rico contributes only the verified dropdown/keyboard questions.
