# Phase 11a — learned cell detector for wide-field malaria (Python only)

**Verdict: the detector clears the recall bar decisively; the full pipeline is a PARTIAL pass.** Detection recall on infected cells went from 30.9% (heuristic) to 93.7% (official test set), and the known-positive Photo 2 is now flagged for the right reasons. But the false-positive gate could not be passed *meaningfully*: on the malaria-free sickle fields the detector is blind at strict thresholds (so a low false-positive rate proves little) and noisy at permissive ones. Photos 1, 3, 4 are still not resolved. No TFLite export or Android work was done (that is Phase 11b, pending your review).

## Scope decision
Single-class "cell locator" as proposed — I agree with the reasoning. Two details: (1) leukocytes are left unlabelled (so the detector learns to skip them; the classifier has a 6.2% WBC false-positive rate), "difficult" boxes count as cells; (2) BBBC041 cells are ~110-140 px native but your photos are 300-500 px with 15-50 px cells, so training used random scale augmentation (native x0.2-0.6) and inference resizes every photo's longer side to 640 px.

## Task 1 — framework check
- **TFLite Model Maker (EfficientDet-Lite): not viable here.** v0.4.3 pins the old TF 2.x / `tf-models-official` stack, incompatible with this machine's TF 2.16 / Keras 3 / Python 3.10 and would break the pinned environment used by every previous phase.
- **Used instead:** a CenterNet-style anchor-free detector in plain Keras (MobileNetV2 backbone, light FPN to stride 4, heatmap + size + offset heads). Only conv / resize ops, so it should convert to TFLite cleanly in 11b (not yet tested); peak picking (3x3 max) is done outside the network.

## Task 2/3 — data and training
`scripts/detector/`. Same split as Phase 9b (seed 42, 85/15 by image from `training.json`; test = official `test.json`). Trained 15 epochs on CPU (58 min), random 384 px crops, flips/transpose, stain jitter, blur.
- **Gotcha, resolved:** for the first ~9 epochs the validation loss was ~11 while train loss was ~1.7, and an early checkpoint found <1% of cells at inference. Cause: stale BatchNorm moving statistics (same crop: 0.93 loss after re-estimating BN vs 11.5 before; weights identical). It self-corrected as the learning rate decayed (val loss 1.28 at the end); the reported results use the final checkpoint with no recalibration.
- Score threshold chosen on **validation only** (highest threshold with val infected recall >= 90%): **0.2**. Nothing tuned on test.

## Task 3 — detection recall (score threshold 0.2)
Match = IoU >= 0.5 (strict); "centre-in-box" is the Phase 9b criterion, for direct comparison with the heuristic.

| Set | All cells | **Infected cells** | Uninfected |
|---|---|---|---|
| Validation, 181 imgs (IoU>=0.5) | 93.4% | **90.9%** (279/307) | 93.5% |
| **Official test, 120 imgs (different microscope)** IoU>=0.5 | 91.0% | **93.7%** (284/303) | 90.9% |
| Official test, centre-in-box | 94.5% | 96.4% | 94.4% |
| **Same 30 test images as the heuristic run**, centre-in-box | 96.2% | **96.3%** (78/81) | 96.2% |
| Heuristic detector, same 30 images | — | 30.9% (25/81) | 55.1% |

Infected cells are found *at least as well* as healthy ones — the failure mode you were worried about did not occur, and it holds on the shifted-domain test set. Cost: ~14-19% of detections do not match any labelled cell (partial cells at borders, WBCs, duplicates).

## Task 4 — full pipeline (detector -> crop x1.15 -> existing BBBC041 classifier, unchanged)
**BBBC041 test images (30 with infected cells, same as the heuristic run), image-level catch:**

| Rule | Detector 0.2 | 0.3 | 0.4 | Heuristic | Oracle boxes (all infected imgs) |
|---|---|---|---|---|---|
| any cell >= 0.5 | **23/30** | 22/30 | 21/30 | 5/30 | 74/115 |
| any cell >= 0.1 | 26/30 | 25/30 | 24/30 | 10/30 | — |
| >= 3 cells >= 0.5 | 1/30 | 1/30 | 1/30 | — | — |

**Your four photos (detector threshold 0.2):**
| Photo | Result |
|---|---|
| **2.jpg** (confirmed positive) | **Flagged.** 50 boxes; two red-cell boxes score 1.00 and 0.69 and are visibly the infected cells (schizont + trophozoite/ring); a third infected cell scores 0.28. Boxes sit on real red cells (`user_photo_2_pipeline.png`, local). |
| 1.webp | 12 boxes, one at 0.53 (a detector-uncertain box) — not reliable |
| 3.jpg | 2 boxes, max 0.21 — effectively nothing found |
| 4.jpg | 7 boxes, one at 0.86 — but the detector's own score for it is low; at detector thr >= 0.3 Photo 4 has 1 box and scores 0.002. Not trustworthy either way. |
Photos 1, 3, 4 are the thick-smear-like / faint-monolayer images identified in Phase 9b; at detector threshold >= 0.3 the detector finds essentially no cells in them, consistent with "no separable red cells". **This confirms rather than fixes the Phase 9b scope limit: thin-smear monolayer photos only.**

**SUPERSEDED as a false-positive test - see "Corrected false-positive test" below. Kept as a separate finding on staining protocols (30 sickle-dataset fields, 1000x1015):**
| Detector thr | cells found | per-cell >= 0.5 | fields flagged, "any >= 0.5" | ">= 3 cells >= 0.5" |
|---|---|---|---|---|
| 0.2 | 1,344 | 1.71% (23) | **13/30** | 3/30 |
| 0.3 | 67 | 1.49% | 1/30 | 0/30 |
| 0.4 | 11 | 0% | 0/30 | 0/30 |
| Phase 9b heuristic | 3,672 | 0.52% | 12/30 | 1/30 |

**Read this as a cross-stain finding, not a specificity test.** At threshold 0.2 the false-positive rate is *no better* than the heuristic pipeline (per-cell 1.71% vs 0.52%). All 23 flagged control cells have detector scores of only 0.20-0.38, i.e. the detector itself is unsure they are cells. At 0.3+ the false positives vanish, but only because the detector then finds 67 (or 11) cells in 30 fields containing ~1,300 — it does not recognise the sickle-dataset stain/appearance as cells, which is out-of-distribution for it. So "0/30 flagged" at 0.3-0.4 says the detector is blind there, not that the pipeline is specific. No conclusion about real-world false-positive rate should be drawn from these controls.

## What this means / recommendations
1. **Detector: good.** Ship-worthy on thin-smear-like data (recall 91-94% on a shifted test set). Ready for TFLite conversion in 11b if you want to proceed.
2. **Pipeline threshold: not chosen.** The sickle-field controls are not a valid basis for it (see below); the corrected same-domain test is in the next section.
3. **Image-level rule:** ">= 3 infected cells" is unusable (1/30); "any cell >= 0.5" catches most positives but a single high-scoring cell is a weak signal (it is also how the controls got flagged). Needs a decision with the false-positive data above.
4. Still true from earlier phases: classifier scores compress under unfamiliar microscope setups; thick-smear photos (1, 3, 4) unsupported; n=4 photos, Photo 2 ground truth is my visual read.

Files: `scripts/detector/{common,train,evaluate,pipeline,recalibrate_bn}.py`; raw outputs in `model_output/detector/{detector_eval,pipeline_eval,pipeline_threshold_sweep}.txt`. Overlay PNGs derived from your photos / third-party images are local-only (gitignored). Model weights: `artifacts/detector/detector_best.keras` (local, gitignored). Nothing committed or pushed.


---

## Known finding (kept separate): the detector does not recognise a different staining protocol
On 30 malaria-free **sickle-dataset** fields (a different stain / camera / 1000x1015 format) the detector finds 1,344 cells at thr 0.2, 67 at 0.3 and 11 at 0.4 (the heuristic found 3,672). It was trained only on BBBC041's Giemsa protocol, so it treats another protocol's cells as not-cells (or as low-confidence cells). This is a real, useful data point — the same cross-domain limitation the app's warning already describes, now also true for the detector — but it says nothing about specificity, so **it is not used to choose a threshold.**

## Corrected false-positive test — genuine malaria-negative BBBC041 fields (same stain as training)
Data availability: BBBC041 has enough for a primary test but not a second one. **Held-out validation split: 30 fields with no infected-stage and no "difficult" box (1,930 labelled cells)** — never trained on by the detector or the classifier. **Official test split: only 3 such fields (97 cells)** — too few for a rate, reported for completeness. "Negative" means annotators labelled no infected cell; unannotated infected cells cannot be ruled out (Phase 9b estimated ~6% of cells are unboxed, and some infected cells may be labelled as ordinary RBCs). Validation images were used earlier to pick the detector threshold via *infected recall*, never for false-positive tuning.

The detector is not blind here: it finds 104-113% of the labelled cells (extras = partial border cells / duplicates).

| Detector thr | Cells found | **Per-cell >= 0.5** (95% CI) | **Fields flagged, any >= 0.5** (CI) | >= 3 cells >= 0.5 | >= 5 cells >= 0.9 |
|---|---|---|---|---|---|
| 0.2 | 2,184 | **1.88%** (1.39-2.54%) | **16/30 = 53%** (36-70%) | 6/30 | 0/30 |
| 0.3 | 2,096 | 1.86% (1.36-2.53%) | 15/30 = 50% (33-67%) | 6/30 | 0/30 |
| 0.4 | 2,003 | 1.60% (1.13-2.25%) | 13/30 = 43% (27-61%) | 5/30 | 0/30 |
| Test split (3 fields, n too small) | 103-120 | 0/103-120 | 0/3 | 0/3 | 0/3 |

**Findings**
1. **The false positives are real within the training domain, and the detector threshold barely affects them** (1.6-1.9% per cell at 0.2-0.4). With ~65 cells per field, a ~1.8% per-cell rate means about half the fields contain at least one cell >= 0.5. So "any single cell >= 0.5" is **not** usable as an image-level rule. The false-alarm source is the **classifier's per-cell rate, not the detector** — expected, since Phase 9b measured 1.1% on sampled validation RBCs.
2. Of the 41 flagged cells at thr 0.2: 26 overlap a labelled red blood cell, 0 a leukocyte, 15 overlap no labelled cell (border fragments / unlabelled). I have not visually reviewed them, so some may be genuine parasites the annotators missed — unverified.
3. The scores of flagged cells in negative fields are often high (0.9-1.0 in ~9 of 16 flagged fields), so raising the per-cell cutoff alone does not fix it either.

**Same-domain rule trade-off** (133 held-out BBBC041 validation fields *with* infected cells vs the 30 negatives; image-level; n=30 negatives means wide intervals, e.g. 6/30 = 20% has a 95% CI of roughly 10-37%):

| Rule | Thr 0.2: infected fields caught / negative fields flagged | Thr 0.4 |
|---|---|---|
| any cell >= 0.5 | 98% / 53% | 97% / 43% |
| any cell >= 0.9 | 94% / 30% | 92% / 30% |
| >= 2 cells >= 0.5 | 81% / 37% | 78% / 33% |
| >= 3 cells >= 0.5 | 60% / 20% | 55% / 17% |
| >= 1 cell >= 0.9 or >= 3 >= 0.5 | 95% / 33% | 92% / 30% |
Nothing on this table reaches a low false-alarm rate (<10%) with high sensitivity; the trade-off is a property of the classifier's per-cell false-positive rate, and the detector threshold moves it only marginally (1-2 fields of 30). For reference, Photo 2 (two cells >= 0.5, scoring 1.00 and 0.69) is caught by every rule above except ">= 3 cells >= 0.5".

Raw output: `fp_bbbc_negatives.txt`, `fp_breakdown.txt`. Scripts: `scripts/detector/fp_bbbc_negatives.py`, `fp_breakdown.py`.
