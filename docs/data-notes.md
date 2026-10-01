# Data source notes

Findings from inspecting the real sources while writing the adapters. The point is to keep the
"labels are known to be true" principle honest. Counts come from hand-checks of screenshots.

## COCO (`data/coco.py`)

- val2017 annotations: 4,807 of 5,000 images have a visible object; 18,536 questions at defaults.
- COCO supercategory names ("outdoor", "accessory", "indoor") are meaningless on their own, so each
  option in `coco.dominant_supercategory` is described by its member categories.
- License: annotations CC BY 4.0; images are Flickr images with individual licenses (about 70% of
  val2017 are NonCommercial). See `docs/data-licenses.md`.

## Rico (`data/rico.py`)

- Use the Hub config `ui-screenshots-and-view-hierarchies` (real screenshots).
- **Do not use** `ui-screenshots-and-hierarchies-with-semantic-annotations`: its "screenshot"
  column is a colour-coded rendering of the component labels, not a real screen. Training on it
  would teach a distribution nobody uses. Its flattened tree also never contains `Text` or
  `Text Button`, so absence there means nothing.
- Hierarchy nodes can be present but invisible, empty or occluded. Visual spot checks on 24
  samples (test shard, 3,312 screens):

  | Check | Result |
  |---|---|
  | Negatives for web view / map / video / switch / dropdown / date picker / slider | 8 of 8 right |
  | Negatives for text input / bottom nav (custom widgets) | 2 of 8 wrong |
  | Positives for ad banner / map / tab bar (first, loose regexes) | 5 of 8 wrong |
  | Positives for ad banner / map / tab bar (visible nodes, exact class names) | 3 of 8 wrong |
  | Positives for dropdown | 2 of 2 right |

- Result: positives are asked only for `dropdown`; the other elements need the teacher check
  (#33, #34) before they are trusted. Negatives are asked only for widgets whose stock class is
  the norm (`NEGATIVES_OK`).
- `is_keyboard_deployed` is an exact label but was `False` for all 3,312 screens in this shard, so
  closed screens are subsampled and open ones are always kept.
- "Screen type" (login, settings, ...) is not labelled in Rico and can't be derived from the
  activity name without noise. Exact screen-type labels come from the synthetic generator (#10).
- Real-screenshot size: about 100 KB per screen (a 322 MB shard holds 3,312).
- License on the Hub card: `unknown`, but the real terms restrict redistribution. See
  `docs/data-licenses.md`.

## RVL-CDIP (`data/rvlcdip.py`)

- Hub repo `aharley/rvl_cdip`: 16 classes; 320k train / 40k val / 40k test; **a single 38.8 GB
  `tar.gz`**. Don't download it whole; stream a subset when building the feature cache (#17).
- License on the Hub card: `other` (scanned tobacco-industry documents from the IIT-CDIP
  collection). Redistribution of the images is not permitted; see `docs/data-licenses.md`.
