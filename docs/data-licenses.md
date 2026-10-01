# Data and model license audit

Status: first pass, 2026-10-01. **This is an engineering audit, not legal advice.** It records what
each source's terms say, how I checked, and what that means for what this project may do. Anything
marked "decision" needs a human before the first public release.

Three separate questions apply to every source:

1. **Train** on it, as an internal experiment?
2. **Publish** its images or labels in a dataset we release?
3. **Publish weights** that were trained on it?

## Summary

| Source | License / terms | Verified from | Publish images | Publish labels / IDs | Weights trained on it |
|---|---|---|---|---|---|
| COCO 2017 annotations | CC BY 4.0 | search summary of cocodataset.org terms | n/a | **Yes**, with attribution | Yes |
| COCO 2017 images | Flickr terms; **per-image license** (below) | primary data (annotation file) | **Only license ids 4, 5, 7, 8** (about 25%) | Yes | **Decision**; safe if filtered to 4, 5, 7, 8 |
| VQA v2 questions + answers | CC BY 4.0 | primary text (visualqa.org/terms.html) | n/a (images are COCO) | **Yes**, with attribution | Yes |
| Rico | University of Illinois terms; screenshots "may contain copyrighted work" | primary text (copyright.txt) | **No** | Unclear: ask | **Decision** |
| RVL-CDIP | Hub license `other`; documents from the Legacy Tobacco library | Hub card + search summary | **No** | Unclear: ask | **Decision** |
| Synthetic web / documents / quality (ours) | ours | n/a | **Yes** | **Yes** | Yes |
| SigLIP 2, ModernBERT-base | Apache 2.0 | Hub card metadata | n/a | n/a | Yes |
| Qwen2.5-VL-7B-Instruct (teacher) | Apache 2.0 | Hub card metadata | n/a | Teacher labels: yes | Yes |
| SmolVLM, moondream2 (baselines) | Apache 2.0 | Hub card metadata | n/a | n/a | n/a |
| External test sets (#22) | not chosen yet | n/a | n/a | n/a | n/a |

## COCO 2017 images: the one that matters most

COCO's annotation file carries a Flickr license for every image. Counting the real `val2017` file:

| License id | License | val2017 images | Share |
|---|---|---|---|
| 1 | CC BY-NC-SA 2.0 | 1,431 | 29% |
| 3 | CC BY-NC-ND 2.0 | 1,414 | 28% |
| 4 | **CC BY 2.0** | 857 | 17% |
| 2 | CC BY-NC 2.0 | 630 | 13% |
| 5 | **CC BY-SA 2.0** | 417 | 8% |
| 6 | CC BY-ND 2.0 | 246 | 5% |
| 7 | **No known copyright restrictions** | 5 | 0.1% |
| 8 | **US Government work** | 0 | 0% |

- About **70% are NonCommercial** (ids 1, 2, 3) and about **33% are NoDerivs** (ids 3, 6).
- **NoDerivs matters beyond redistribution.** The blur, noise, JPEG and brightness generator (#12)
  produces modified copies. A modified copy of an ND image is a derivative.
- Ids 4, 5, 7 and 8 permit sharing and modification (with attribution, and share-alike for id 5).
  They are about 25% of COCO, which still leaves tens of thousands of train images.
- COCO states that it does not own the images and that users accept responsibility for how they
  use them (via the Flickr terms).

**Recommendation (implemented as an opt-in):** build every dataset that will ship with
`iter_coco(..., licenses=PERMISSIVE_LICENSE_IDS)`. It is off by default so existing experiments
don't change silently. Never publish COCO images themselves; publish IDs and labels, and let users
fetch images from COCO. Each COCO fact now records its `license`.

Whether training on NonCommercial images and releasing Apache-2.0 weights is allowed is legally
unsettled. Filtering to ids 4, 5, 7, 8 avoids the question.

## VQA v2

The questions and annotations are CC BY 4.0 ("The annotations in this dataset belong to the VQA
Consortium and are licensed under a Creative Commons Attribution 4.0 International License").
The images are COCO's, so the COCO image rules apply. Our adapter keeps only the yes/no answers,
which may be published with attribution.

## Rico

Verbatim from the dataset's `copyright.txt`:

> The screenshots contained in the Rico dataset may contain copyrighted work. ... Researcher
> accepts full responsibility for his or her use of the Database and shall defend and indemnify the
> Rico team and the University of Illinois ... against any and all claims arising from Researcher's
> use of the Database, including but not limited to Researcher's use of any copies of copyrighted
> images that he or she may create from the Database. ... Researcher may provide research
> associates and colleagues with access to the Database provided that they first agree to be bound
> by these terms and conditions. ... If Researcher is employed by a for-profit, commercial entity,
> Researcher's employer shall also be bound by these terms and conditions.

- **Do not publish Rico screenshots or any copies of them.** Publishing would hand the database to
  people who haven't agreed to these terms.
- Our questions refer to screens by ID only, but the labels come from Rico's view hierarchies,
  so even a labels-only release is a gray area. Ask the Rico team before publishing it.
- The terms say nothing about trained models. The indemnity wording makes this a **decision**.
- The Hub card says `unknown`, which is wrong to rely on: the real terms above exist.
- Rico's verified contribution is small (only the dropdown question and the keyboard flag; see
  `docs/data-notes.md`), so leaving it out of the training mix for released weights costs little.

## RVL-CDIP

- The Hub card gives license `other` and says: "RVL-CDIP is a subset of IIT-CDIP, which came from
  the Legacy Tobacco Document Library, for which license information can be found here."
- The Industry Documents Library (UCSF) copyright page says each user of the site is responsible
  for complying with copyright law. It has a take-down policy and fair-use guidance, and it does
  not grant a blanket reuse license.
- **Do not publish the images.** The set is a single 38.8 GB archive, so it is also heavy to host.
- Publishing class labels per document ID is unclear (Harley et al. produced the labels). Ask.
- Whether to train released weights on it is a **decision**. It adds 16-way real-document variety
  that our three synthetic document types can't, so leaving it out has a real cost.

## Synthetic data (ours)

Web pages, documents and quality-degraded copies of **synthetic** images are generated by this
repository's code and may be published, for example under CC BY 4.0 or Apache 2.0.
Rendering uses system fonts (Helvetica, Georgia, Courier New, Verdana, Trebuchet MS). We don't
distribute the fonts, only pixels drawn with them; this is normally unrestricted, but it is the one
third-party element, so it is listed. Quality-degraded copies of **third-party** images inherit the
source's restrictions.

## Models

Read from the Hugging Face model cards' license metadata:

| Model | License |
|---|---|
| `google/siglip2-base-patch16-384` | Apache 2.0 |
| `answerdotai/ModernBERT-base` | Apache 2.0 |
| `Qwen/Qwen2.5-VL-7B-Instruct`, `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` | Apache 2.0 |
| `HuggingFaceTB/SmolVLM-Instruct` | Apache 2.0 |
| `vikhyatk/moondream2` | Apache 2.0 |

Use the 7B Qwen2.5-VL as the teacher. Some other Qwen-VL sizes use a different license, so don't
swap sizes without re-checking. Apache 2.0 requires keeping license notices in released weights.

## External test sets (#22)

Not chosen yet. A candidate must (a) permit evaluation use and (b) have terms recorded in this file
before it is added. Evaluation-only data is never redistributed unless its license allows it.

## Open decisions for the first public release

1. **COCO:** adopt the license filter for everything that ships? *Recommended: yes.*
2. **Rico:** include in the training mix for released weights, or ask the Rico team first?
   *Recommended: leave it out of the released weights' mix, keep it for internal experiments.*
3. **RVL-CDIP:** include in the training mix for released weights?
   *Recommended: internal experiments only until permission or clearer terms; if that hurts the
   document benchmark, say so in the model card.*
4. **Dataset license** for the synthetic data: Apache 2.0 or CC BY 4.0.

These default to the conservative choice, because a model can be re-released with more data later
but a release made with unclear rights cannot be taken back.
