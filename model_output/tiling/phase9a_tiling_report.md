# Phase 9a — Cell-level tiling for malaria: go/no-go checkpoint (Python only)

**Status: NO-GO for tiling as designed.** The sections below were written first, from sickle-dataset fields only; the four user-supplied malaria-positive photos were then run (see "Update" at the end) and the key pass/fail check failed. No Android code was touched.

## What was built
`scripts/tiling/` — `detect_cells.py` (OpenCV: green channel + CLAHE → field mask → adaptive threshold → morphology + hole fill → distance-transform watershed → per-blob filtering), `classify_tiles.py` (square crop around each kept cell → 224×224 → existing malaria **Keras** model, same `(x-127.5)/127.5` preprocessing), `run_detection.py`, `run_tiling_eval.py`. Otsu thresholding was tried first and rejected (merges clumps); adaptive thresholding is used. A Hough-circle comparison was also tried and abandoned: it returned 0–27 circles per image with no relation to the true cell count.

## Task 1 — detection quality (no cell-level ground truth exists; judged from statistics + overlays)
Evaluated on 12 sickle-dataset field images (6 "Clear" round-cell fields, 6 sickle-positive fields), then classification on 44.

| Field type | Result |
|---|---|
| Dense, in-focus, round, touching cells (most "Clear" fields; e.g. `clear_10`: 287 blobs, 278 kept, only 1 blob >2× median area, kept-area CV 0.17) | **Usable.** By eye nearly every cell is individually outlined; a few percent of cells at clumps/edges merged or missed. |
| Sparse/faint, clumped, irregular or sickled cells (`sickle_pos_1`, `_100`, `_102`) | **Unreliable.** 14–69 blobs >2× median (merged clumps, which are discarded and therefore *never classified*), 28–47 fragments <0.45×, and many faint cells missed. Only ~50% of blobs survive filtering (1,191 of 2,186 across the 14 sickle fields). |
| Out-of-focus (`clear_102`) | **Fails.** 235 blobs → 62 kept; outlines follow blur, not cells. |

Estimated false splits / false merges: **not measurable exactly** (no ground truth; I did not hand-count). Proxies: in good fields ~1–5% of blobs are merges (>2× median area); in clumped fields 15–40% of blobs are merges, and merged clumps are dropped rather than split, so cells inside clumps are silently absent from the result. Over-splitting is minor by eye (tiny fragments are filtered by area) but the fragment count is high in sickle fields.

**Segmentation is workable only on dense, focused, round-cell fields.** Whether that matters depends on what a real malaria photo looks like — which is why the positive photo is needed.

## Task 2 — per-cell classification (malaria Keras model)
Baseline: 44 field images from the sickle dataset (30 Clear + 14 sickle-positive), 5,502 cells. These are **assumed malaria-free but not certified** as such (sickle-cell dataset, not a malaria dataset). The whole-image malaria model gives P ≤ 0.016 on all 44, so they are a fair false-positive baseline for the tiling approach.

Two crop styles were compared, because the malaria model was trained on cells segmented onto a **black** background:

| Crop style | per-cell P≥0.5 | P≥0.9 | P≥0.99 |
|---|---|---|---|
| **raw** square crop (light background + neighbours) | 11.4% | 4.5% | 1.45% |
| **masked** (pixels outside the detected cell set to black) | 7.9% | 0.6% | 0.11% |

- **Raw crops are unusable.** Healthy red cells score 0.99–1.00, a white blood cell scores 1.00. The model reacts to the background/neighbour context, not parasites (`cmp_top10.png`, top panel).
- **Masked crops are better but still noisy.** The top-scoring "positive" crops on malaria-free fields are platelets touching a red cell (0.72–0.88), a WBC (0.66), stain debris (0.58) and notched / mis-segmented cells — exactly the look-alikes of a parasite. (`cmp_top10.png`, bottom panel.)

### Field-level false-positive rate of candidate aggregation rules (masked crops, 44 malaria-free fields)
| Rule | Malaria-free fields wrongly flagged |
|---|---|
| any cell ≥ 0.5 | 93% |
| any cell ≥ 0.9 | 48% |
| ≥ 3 cells ≥ 0.5 | 82% |
| ≥ 10% of cells ≥ 0.5 | 30% |
| ≥ 20% of cells ≥ 0.5 | 14% |
| **≥ 5 cells ≥ 0.9** | **2%** |

The "count positives / show the top few most-positive crops" style of aggregation **does not work as-is**: a malaria-free field routinely has a top crop at 0.85–0.99. Only a strict rule (≥5 cells ≥0.9) has an acceptable false-positive rate here, and **its sensitivity is unknown** — a real positive field must produce ≥5 such crops through this same noisy segmentation, which one photo cannot establish.

## Task 3 — honest assessment
- **Segmentation noise does not "wash out."** Failures are systematic, not random: clumped/faint/blurred fields lose cells (merges dropped), and the dominant classification error is *artifact look-alikes* (platelets, WBCs, debris), which appear in every field and are not averaged away by having more cells — more cells means more artifacts.
- The pipeline is **fragile** to focus, stain and clumping, and the classifier-side problem is larger than the segmentation-side problem. Every image tested has the sickle-dataset staining/format, not thin-smear malaria photos.
- Whole-image classification currently gives 0/44 false alarms on these fields; tiling with the current model is *strictly noisier* unless a strict rule + a positive-field sensitivity check justifies it.

## Go/no-go and what I need
**Cannot call it yet.** To decide I need (1) your known-positive field photo, and ideally **several** positive fields — one image cannot measure sensitivity or set a threshold; (2) optionally, thin-smear malaria field images with cell annotations (the NIH/NLM thin-smear field set is the natural source; I have not verified its availability). If the positive photo passes the strict rule, the remaining risk is the artifact false-positive rate. If it does not, the likely fix is not more tuning but retraining the malaria model with hard negatives (platelets, WBCs, debris cropped from fields) — a retraining decision I'd ask you about first.

Example figures: `example_overlays.png` (green = kept cell, red = merged/too large, blue = tiny fragment, orange = edge/non-convex), `top_crops_negative_field.png`.

---

## Update: the four user-supplied malaria-positive photos
Photos: `1.webp` (400×300), `2.jpg` (410×308), `3.jpg` (447×447), `4.jpg` (509×510) — small web-resolution images. Figures: `user_photos.png`, `user_photo_overlays.png`, `user_photo_top_crops_1_3_4.png`, `user_photo_2_crops.png`.

| Photo | What it is (by eye) | Whole-image P(malaria) | Cells detected | Tiling: masked crops ≥0.9 / ≥0.5 | Strict rule (≥5 cells ≥0.9) |
|---|---|---|---|---|---|
| 1.webp | dense smear with lysed background + WBCs (thick-smear-like) | 0.463 | 52 (mostly stain patches/WBCs) | 9 / 32 | flags — for the wrong reason |
| 2.jpg | clean thin smear, ~30 RBCs, several visibly infected cells, watermark | **0.879 (correct)** | **7 of ~30** | 0 / 1 | **does not flag** |
| 3.jpg | very sparse, tiny dots (low-magnification / thick-smear-like) | 0.162 (miss) | 79 (mostly dots/debris) | 6 / 31 | flags — for the wrong reason |
| 4.jpg | dense faint smear, WBCs, a green annotation arrow | **0.011 (miss)** | 60 (stain patches, WBCs, the image border) | 11 / 47 | flags — for the wrong reason |

**Result: the tiling approach fails the pass/fail check.**
- **The one photo that matches the intended input (2.jpg) is already classified correctly by whole-image inference (0.88).** Tiling adds nothing there and does worse: the detector finds only 7 of ~30 cells and only ~1 of the ~4 visibly infected ones (faint cells on a white background were missed), so the strict rule does not fire.
- **On the photos where whole-image missed (3, 4) or was uncertain (1), tiling "finds" positives, but they are not real signal.** The crops it scores are stain texture, background patches, WBCs, and black-edged blocks at the image border (`user_photo_top_crops_1_3_4.png`) — none are single red cells. The same behaviour (score ≥0.9 on non-parasite material) produced the false alarms on malaria-free fields, so a rule that "passes" here cannot be told apart from a false alarm.
- **These photos are mostly not the format the pipeline assumes** (well-focused thin-smear monolayer of separable red cells at usable resolution). 1 and 3 look like thick-smear/low-magnification images; 4 has faint, overlapping cells and an annotation arrow; all are ≤510 px, so a cell is ~15–30 px before being enlarged to 224 (the model was trained on ~124 px cells).
- Caveats: four images, ground truth taken from the user's statement that all are malaria-positive plus my visual read of photo 2 (I am not a parasitologist); no per-cell labels.

**Recommendation: do not build tiling into the app (no-go for 9b as designed).** The real problem with these photos is that the malaria model doesn't transfer to other stain/format/scale (photos 3 and 4), which cell tiling does not fix and in fact hides behind artifact-driven positives. Options: (a) leave the whole-image malaria check and state clearly what image type it supports; (b) collect/label a small set of these field-style malaria photos and retrain or fine-tune (a data + retraining decision I would need your go-ahead on); (c) if wide-field support is a hard requirement, source thin-smear field datasets with cell-level labels before more prototyping.


*Figures derived from the user-supplied photos and from BBBC041 images (`user_photo*.png`, `complete_A.png`, `photo2_oracle.png`, `photo_overlays_new.png`) are kept local and not committed, because their redistribution terms are unverified. The scripts in `scripts/tiling/` and `scripts/bbbc041/` regenerate them.*
