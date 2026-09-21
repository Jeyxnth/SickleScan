# Phase 9b — BBBC041 validation for field-photo malaria detection (Python only)

**Verdict: PARTIAL. BBBC041 fixes the classifier-side problem from Phase 9a (artifact-driven false positives, scale) but does NOT fix the pipeline: the heuristic cell finder is the bottleneck, the score calibration does not transfer across imaging setups, and three of your four photos are a different preparation type that thin-smear data cannot address. Do not commit to a full retrain/integration yet.** No Android code was touched.

## Task 1 — data audit
BBBC041 (P. vivax, Giemsa thin smears): 1,328 images, `training.json` 1,208 images (1600×1200 PNG) + `test.json` 120 images (**1944×1383 JPG — a different imaging setup**). Boxes are `{minimum{r,c}, maximum{r,c}, category}`.

| Class | Train boxes | Test boxes | Median box (train / test) |
|---|---|---|---|
| red blood cell | 77,420 | 5,614 | 108 px / 139 px |
| trophozoite | 1,473 | 111 | 131 / 165 |
| ring | 353 | 169 | 121 / 153 |
| schizont | 179 | 11 | 142 / 152 |
| gametocyte | 144 | 12 | 131 / 164 |
| leukocyte | 103 | **0** | 119 |
| difficult | 441 | 5 | 133 |

- Infected ≈ 2.7% of boxes (train), 5.1% (test): strongly imbalanced. **Test has no leukocytes**, so white-cell false positives can only be measured on validation.
- **Scale:** ~108–139 px cells at native resolution, i.e. the same order as the NIH crops. The Phase 9a "scale mismatch" is really a mismatch with the *user photos*, which are 300–500 px images with 15–50 px cells.
- **Annotation completeness (20 random training images, overlays in `complete_A.png`):** by eye nearly every whole red cell is boxed and infected cells are marked; unboxed items are mostly partial cells cut by the image border. Proxy: only **6.4%** of the heuristic detector's confident detections (54/845) fall outside every box. Some cells carry duplicate/overlapping boxes. So unboxed regions can be treated as background with modest risk (I did not sample background patches; I used boxes only). Staining varies a lot across slides (dark purple to almost greyscale), which is useful for generalisation.
- Bonus: real ground truth finally lets the Phase 9a detector be scored. On these 20 images it missed **35%** of annotated cells (476/1,344), up to 119/171 in dense images.

## Task 2 — classifier from native-scale crops
Crops = box × 1.15, native resolution → 224×224; infected = ring/trophozoite/schizont/gametocyte, not infected = RBC + leukocyte (hard negatives); "difficult" excluded. **Splits by image** (train 1,018 / val 181 images from `training.json`; test = official `test.json`). RBCs subsampled (train 6,000, val 1,500). MobileNetV2 → GAP → Dense128 → Dropout → sigmoid, head then fine-tune top 30 layers, augmentation includes stain (hue/saturation) jitter, random grayscale and downscale→upscale blur.

| Evaluation | Result |
|---|---|
| Validation (held-out images, same setup as training) | AUC 0.998, infected recall **97.1%**, RBC false-positive 1.1%, **leukocyte false-positive 6.2%** (n=16) |
| **Official test set (different setup), GT boxes as oracle detector** | AUC **0.985**, but sensitivity **40.9%** at 0.5 (specificity 99.9%, 5 FP of 5,614) |
| test, per stage recall @0.5 | ring 30%, trophozoite 54%, schizont 73%, gametocyte 42% |
| test, threshold retuned | at FPR ≤1%: sensitivity **84.8% but only at threshold 0.013**; at 0.1: 67%/99.7% |

**Reading:** the model ranks infected above healthy cells well even on the shifted test set (AUC 0.985), but its *scores are compressed* under a different microscope/camera setup (median infected score 0.33). A fixed 0.5 threshold is therefore not portable — and every phone/microscope combination is another unseen setup. Nothing here was tuned on the test set.

### Task 2.2 — learned detector
**Not built.** No detector framework is installed and a TFLite-friendly one (e.g. SSD-MobileNet/YOLO-nano) is a heavy new dependency I would ask you about first. What I measured instead shows why it is needed: on 30 annotated test images the Phase 9a heuristic finds only **30.9% of infected cells** (25/81) versus 55.1% of healthy cells (755/1,371) — infected cells are darker/irregular and get filtered out. With that detector the **full pipeline catches infected images only 5/30 (rule "any cell ≥0.5")**, 10/30 at ≥0.1, versus 74/115 with oracle boxes.

## Task 3 — your four photos (pass/fail check)
| Photo | Old NIH model, whole image | NEW: heuristic detector + BBBC classifier | Notes |
|---|---|---|---|
| 1.webp | 0.463 | 52 crops, 0 ≥0.5 (max 0.04) | lysed/thick-smear-like; not a red-cell monolayer |
| 2.jpg | 0.879 (correct) | 7 crops, **0** ≥0.5 (max 0.004) | detector missed every infected cell |
| 3.jpg | 0.162 (miss) | 79 crops, 0 ≥0.5 (max 0.02) | very sparse tiny dots |
| 4.jpg | 0.011 (miss) | 60 crops, 1 ≥0.5 (0.57) | thick-smear-like; arrow points at an ambiguous faint patch |

**Oracle check on photo 2 (my hand-marked boxes, not lab truth; `photo2_oracle.png`):** infected cells score **1.00 (schizont), 0.78, 0.54; all 10 healthy cells score ≤0.002.** So the classifier *does* recognise real infected cells from a small foreign web image, and does not fire on healthy ones — the pipeline failed only because the detector never handed it the infected cells.

**Controls (30 malaria-free sickle-dataset fields, same pipeline, 3,672 cells):** per-cell ≥0.5 **0.52%** (was 7.9% with the NIH model); rules: "≥5 cells ≥0.9" 0/30, "≥3 cells ≥0.5" 1/30, "any ≥0.5" 12/30 — artifact-driven false positives are largely gone.

### Answers to the pass/fail questions
- *Does the known-positive photo now score positive on real cell regions rather than artifacts?* **No.** Photo 4 (the confirmed false negative) is still negative, and so are 1 and 3 — but now for a clearer reason: 1, 3 and 4 are lysed / thick-smear-like preparations with no separable red cells, so a per-red-cell classifier has nothing to score. BBBC041 is thin-smear data and cannot fix that; parasites there are small dots, not infected cells. Only photo 2 is the right kind of image, and there the classifier works but the detector fails.
- *Do the negative-leaning photos stay negative?* None of your four is a negative; the artifact-driven "positives" from Phase 9a are gone (0 flagged in photos 1 and 3, 1 weak in photo 4, 0.5% on controls).

## Recommendation
1. **Do not start a full retrain/integration on this evidence.** The classifier works; three blockers remain: (a) a learned cell detector is needed (heuristic recall on infected cells 31%), (b) calibration shifts between imaging setups (0.5 threshold → 41% sensitivity on the shifted test set), (c) thick-smear-style photos need a different approach/data.
2. Options for you: **(A)** build a small learned detector on the BBBC041 boxes (new dependency — your call) and re-run this exact validation; **(B)** target thin-smear photos only and state that scope plainly in the app; **(C)** collect a few dozen of your own labelled photos to measure/calibrate the score threshold per setup before trusting any fixed number.
3. Caveats: n=4 photos; photo 2 ground truth is my visual read; BBBC041 is P. vivax and the test set is the only shifted-domain evidence; no result here was tuned on the test set.

Figures: `complete_A.png` (annotation overlays; magenta = detected cell with no box), `photo2_oracle.png`, `photo_overlays_new.png` (red = crop scored ≥0.5).


*Figures derived from the user-supplied photos and from BBBC041 images (`user_photo*.png`, `complete_A.png`, `photo2_oracle.png`, `photo_overlays_new.png`) are kept local and not committed, because their redistribution terms are unverified. The scripts in `scripts/tiling/` and `scripts/bbbc041/` regenerate them.*
