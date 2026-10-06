# v0 results: what was built, what was found, what is not yet true

Status as of the first complete benchmark (bench v1). Numbers are in `results/README.md`; this page
is the interpretation, written to be read by someone deciding what to do next.

## The claim being tested

A small model that makes typed decisions (choice, score, yes/no) about an image in one forward
pass, with calibrated confidence, and that **follows written criteria it has never seen** better
than the obvious cheap alternatives (SigLIP 2 zero-shot, a 500M VLM).

## Verdict: not yet

On task families never seen in training (`test-tasks`) the pilot model is **worse** than SigLIP 2
zero-shot (accuracy 0.346 vs 0.426, ECE 0.148 vs 0.069), so release criteria 1 and 2 fail and,
by the rule in the spec (section 7.4), **no model weights are published**. What *is* true:

- On tasks it trained on, with unseen images or unseen page/document styles, it is clearly better:
  accuracy +4 to +12 points in every domain and lower log loss everywhere.
- It runs one image in about 45-75 ms cold and about 6-11 ms for each extra question on the same
  image; SigLIP zero-shot is about 1 ms warm and the 500M VLM about 70-110 ms warm.
- On real external data (Oxford Pets, CORD-v2 receipts, ScreenSpot) it is marginally better than
  SigLIP (0.875 vs 0.867), within noise for the smaller domains.

## What the experiments showed

1. **A fusion model cannot learn text-image alignment from scratch in useful time.** The first
   pilot (ModernBERT text features + SigLIP image features, no prior) kept yes/no loss at ln 2 for
   500 steps. ModernBERT's and SigLIP's features were never trained together. SigLIP's own text
   tower is aligned, which is why its zero-shot yes/no score has signal (AUROC 0.69).
2. **So the model now starts from SigLIP's zero-shot logits and learns a correction** (zero-initialised
   head; a fresh model makes the same decision as the baseline on 800 of 800 real questions).
   Yes/no then learns, and accuracy on trained task families rises quickly (seen-task choice
   0.76 -> 0.97 by step 500).
3. **But the correction memorises task templates.** Seen-task accuracy keeps rising while
   unseen-family performance does not improve past about step 250. `val-tasks` accuracy on choice and
   score is slightly *below* the raw prior, while yes/no is above it. The likely mechanism
   (hypothesis, not tested): each domain has very few task families per question type, so the model
   learns "this domain's score questions are about blur" and applies that to a new score question.
   Screenshots have one training score family and fall from 0.347 to 0.170 on the held-out one.
4. **The 500M VLM is not a strong baseline here.** On identical images it scores 0.538 on
   `test-images` (SigLIP 0.585) and 0.485 on external data (SigLIP 0.868).

## Things that went wrong and were fixed (all caught by checking, none by luck)

| Problem | How it was found | Effect had it shipped |
|---|---|---|
| Rico screens shared one image per non-unique `request_id` | feature cache had fewer images than the facts file | most Rico labels attached to the wrong screenshot |
| Image subsampling used the same hash as the split builder | VLM scored 0.24 on a half-yes/no benchmark | a "random" subset was 100% quality-degraded copies |
| Soft (teacher) labels were silently discarded by the templater | reading the code before the first teacher run | the teacher slice would have trained as hard labels |
| No guard stopped teacher labels reaching evaluation splits | design review | soft labels graded as ground truth |
| COCO: 70% of images are NonCommercial, 33% NoDerivs | read the per-image license field | a release trained on and republishing NC images |
| Pooling over instruction tokens made options indistinguishable | a toy learnability test that failed | choice stuck at chance |
| `fit_platt` Newton steps diverged on badly scaled scores | a recovery test against the analytic answer | wrong calibration for every yes/no question |

## What is not done

- **#33** the Qwen2.5-VL-7B teacher run (the machinery is tested; the run needs a 15 GB download and
  hours of GPU time).
- **#34** the hand-check of teacher labels (needs a person).
- **#35 / #36** the remaining ablations (prior on/off, LoRA, text encoder) and a long final run. The
  `correction_l2` sweep and the extra task families are done (see the update below).
- **#30** nothing has run on Kaggle; **#20** no T4 latency.
- **#24** the accuracy bar for criterion 3 awaits an owner decision (a proposal is in the spec).
- Release criterion 4 (option-shuffle invariance) is guaranteed and tested at the model level for
  choice and bool, but has not been run as a benchmark on the trained model.
- **#38-#42** naming and publishing need the owner's decisions and credentials.

## Update: the first two next steps were tried (pilots v2, v3, v4)

Both steps from the list below were run on bench v2 (bench v1 plus eight fact-based task families,
#86), all on the same 2,495-question `test-tasks` set; full numbers in `results/README.md`.

| model | accuracy | log loss | ECE |
|---|---|---|---|
| SigLIP 2 zero-shot | 0.426 | 1.153 | 0.069 |
| pilot v1 (no extra families, no shrinkage) | 0.346 | 1.251 | 0.148 |
| pilot v2 (more task families) | 0.354 | 1.285 | 0.184 |
| pilot v3 (+ `correction_l2` = 1.0) | 0.425 | 1.133 | 0.072 |
| pilot v4 (`correction_l2` = 0.3) | 0.396 | 1.226 | 0.177 |

- **More task families alone did not help** (v2 is no better than v1 on unseen families).
- **Shrinking the correction helped a lot** (v3 recovers to parity with SigLIP: accuracy level, log
  loss slightly lower), which supports the "memorised templates" diagnosis. But **parity is not a
  win**, so criteria 1-2 are still not met. One seed per row; the selection metric has about 0.02 of
  seed noise.
- **`val-tasks` cannot tune the shrinkage.** l2 = 0.3 was better than 1.0 on `val-tasks` and worse
  on `test-tasks`. With 1,165 questions and few families, `val-tasks` picked the wrong setting.
- Net: the learned correction adds nothing on unseen families once it is constrained enough not to
  hurt, and everything it learns helps only on seen families. That is the evidence for moving to
  approach B rather than tuning approach A further.

## What to try next, in order (original list; 1 and 2 are done, see above)

1. **More task families per (domain, question type).** The diagnosed failure is too little variety.
   Generating several more score and choice families per domain from the facts already stored is
   cheap, and the teacher slice adds ~25 judgement criteria on real images.
2. **Shrink the correction** (`correction_l2` > 0, dropout): cheap, already implemented, and aimed at
   exactly this failure. Compare on `val-tasks`.
3. **Longer is not the answer:** more steps raised seen-task accuracy and left unseen tasks flat or
   worse. Select early (the pilot's best was step 250).
4. If 1-3 do not move `test-tasks`, the spec's fallback is **approach B** (a small VLM used as the
   encoder, answers read from logits), which brings real text-image alignment instead of a prior.

## Decisions waiting on you

- Whether to keep investing in approach A or switch to approach B. **Recommendation: switch to B**
  (open the approach B epic, #42), for the reasons in the update above.
- The criterion-3 accuracy bar (#24).
- The final project name (#38) and whether to publish the benchmark and this write-up now.
