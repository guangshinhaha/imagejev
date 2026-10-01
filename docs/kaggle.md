# Training on Kaggle

The free Kaggle GPU (a T4, up to 9 hours per session, about 30 GPU-hours a week) is where the longer
runs go. A run is a chain of sessions: each trains until a time budget, saves a checkpoint and
stops; the next session resumes from it.

**Status:** the code, config and notebook are tested locally (budgeted sessions chained on CPU give
the identical result to one uninterrupted run), but **nothing has been run on Kaggle itself yet**.
The first real session is the real test; the points most likely to need adjusting are fp16
autocast on CUDA (untested here, only bf16 on CPU is) and the exact input paths Kaggle assigns.

## Once: build and upload the data

On your own machine, after building the benchmark data and the feature cache
(`python -m imagejev.data.build ...` then `python -m imagejev.cache ...`):

```bash
python - <<'PY'
from imagejev.train.kaggle import package_dataset
print(package_dataset("data/bench_v0/splits", "data/cache_v0", "kaggle_data",
                      "YOUR_KAGGLE_USERNAME/imagejev-data", "imagejev data"))
PY
kaggle datasets create -p kaggle_data        # needs ~/.kaggle/kaggle.json
```

- The dataset is created **private**. Keep it private: the features and labels derive from sources
  with restrictive terms (`docs/data-licenses.md`).
- Size: about 100 KB per image, so 13k images is 1.3 GB. The limit for a dataset is far above that.

## Each session

1. Create a Kaggle notebook from `notebooks/kaggle_train.ipynb`, named `imagejev-train`.
2. Turn on a **GPU** and **Internet**, and attach the dataset `imagejev-data` as an input.
3. From the second session on, also attach **the previous version of this notebook** as an input
   (Add Input -> Notebook Output). The notebook restores `last.pt` from it automatically.
4. **Save Version -> Save & Run All (Commit)**. It trains for `time_budget_minutes` (500 by
   default), saves `last.pt` and stops. Its output is kept for the next session.
5. Repeat until the notebook prints `done at step 3000`.

Resuming is exact: weights, optimizer, scheduler, step counter and every random-number generator
are restored, so a chain of sessions gives the same weights as one long run (tested).

## Notes

- `ckpt_every: 100` keeps at most 100 steps at risk if a session dies without warning.
- fp16 autocast with a gradient scaler is used on the GPU (`amp: fp16`); evaluation uses autocast
  too. If you see NaN losses, set `amp: none` first.
- The first session downloads ModernBERT from the Hugging Face Hub (about 600 MB), which needs
  Internet to be on.
