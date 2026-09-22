> **Superseded by `phase14b_guardrail_v3_results.md` (wide-field + single-cell version, the one shipped).**

# Phase 14 — guardrail retrained with BBBC041 wide-field images (Task 1 results; integration NOT started)

**Summary: the retrain fixes the false rejections completely and does not regress on sickle-cell or NIH single-cell images, but it gives a small increase in false acceptances on genuinely unseen non-smear images (0.10% -> 0.31%, not statistically significant), concentrated in dot/blob-pattern textures. The exact Phase 13 non-smear sample shows no change, but that set is mostly training data and proves little. Stopped here for your decision before any integration code was touched.**

## What was trained
Same recipe, code, seed, augmentation, head (dense128), class weights and early stopping as Phase 7 (`scripts/guardrail/train.py`); only the data differs. Smear class + **822 BBBC041 wide-field images (training split only)**; 205 more BBBC041 training-split images used for early stopping. **Never in training:** the 181 held-out BBBC041 validation images (they contain the 163 fields used in Phases 11-13; enforced by assertion) and the 120 official BBBC041 test images. All old rows and splits unchanged. Train 5,440 (smear 3,026 / not-smear 2,414). Scripts: `scripts/guardrail_v2/`. Threshold unchanged at P(smear) >= 0.5.

## Task 1 validation (v1 = the bundled Phase 7 model, v2 = retrained; same images for both)
**1. BBBC041 wide-field images (should now ACCEPT):**
| Set | v1 accepted | **v2 accepted** |
|---|---|---|
| 163 held-out validation fields (same lab/setup as training data) | 1/163 (0.6%) | **163/163 (100%)** [97.7-100%], min P(smear) 0.945 |
| 120 official test images (**different microscope**, never used) | 56/120 (46.7%) | **118/120 (98.3%)** [94.1-99.5%] |

**2. Existing smear sets (no regression):**
| Set | v1 | v2 |
|---|---|---|
| Sickle-cell test images (held out) | 86/86 | 86/86 |
| NIH single-cell malaria test images (held out) | 387/387 | 387/387 |

**3. Non-smear images (should REJECT):**
| Set | v1 rejected | v2 rejected |
|---|---|---|
| Phase 13 sample: 360 natural (coco, documents, dtd, lfw, places, screenshots) + 240 synthetic capture failures = 600 | 600/600 | 600/600 |
| ...of which the 360 natural only | 360/360 | 360/360 |
| All HELD-OUT real non-smear in the guardrail data (val + test, 945) | 944/945 (99.9%) | 943/945 (99.8%) |
| Held-out synthetic capture failures (90) | 90/90 | 90/90 |
| **986 FRESH non-smear images never used anywhere** (250 COCO, 250 Places, 200 DTD, 200 LFW, 86 screenshots; near-duplicates of existing negatives removed) | 985/986 (99.9%) | **982/986 (99.6%)** |
**Important about the Phase 13 sample:** 420 of its 600 images (70%) were in the guardrail's own *training* split (val 96, test 84), so its 100% says little for v2 or v1. The held-out and fresh rows are the real generalisation evidence.

## Where v2 is worse: the 6 false accepts on unseen non-smear (of 1,931)
Combined held-out real + fresh: **v2 accepts 6/1,931 (0.31%) vs v1 2/1,931 (0.10%)**; Fisher exact p = 0.29 for all 1,931, so the increase is **not statistically distinguishable from chance**, but it is in the direction you asked me to guard against. (v1's own measured false-accept rate in Phase 7 was 1/517 = 0.19%.) All six are dot / blob patterns that resemble a field of cells (`phase14_false_accepts.png`, local): polka-dot wall, ink splatter, red spots, foam bubbles, a plate on a red cloth, a blue swirl fabric. v2 P(smear): 0.98, 0.97, 0.91, 0.87, 0.64, 0.58. Five of the six are DTD textures the old model rejected (its scores 0.00-0.45); the sixth (the plate on a red cloth, COCO) was accepted by the old model too (0.977).
**What the rest of the wide-field pipeline does with those six** (detector + classifier + 20-cell gate): none becomes a Positive; 3 are stopped by the gate (Inconclusive: 16, 15, 9 cells), 3 would show a **Negative** verdict (60, 90, 67 cells; top cell scores 0.17, 0.73, 0.91). So the cell gate catches half of what slips through; the other half would display a misleading Negative.

## Other findings
- **Single-cell malaria mode with BBBC041-style crops is still rejected by the guardrail (both versions):** only 7.9% (v1) / 8.4% (v2) of the 1,823 BBBC041 single-cell validation crops are accepted. The guardrail's single-cell positives are NIH-style crops; the Phase 10 single-cell classifier is BBBC041-trained. This matches the phone's log (many "overridden" single-cell sessions). It is outside what you asked me to change; adding BBBC041 single-cell crops as smear examples would be the analogous fix, and needs your go-ahead.
- Float16 TFLite export of v2: 4.79 MB, 1/2,777 decisions differ from Keras (a near-threshold image, max probability difference 0.026); float32 exact. Not copied into the app.
- Caveats: the 163 fields come from the same lab and microscope as the training images, so 100% there is expected; the 120-image official test set (different microscope) is the more informative generalisation number (98.3%). The non-smear evidence is public-dataset photos; real phone photos through a microscope attachment are untested. The threshold 0.5 was not tuned (and moving it would not cleanly help: v2's lowest BBBC field score is 0.945).

## Decision for you
A. **Proceed with integration (my recommendation):** false rejections of wide-field smears go from 99.4% to 0%, at a cost of about 0.2 extra false accepts per 100 non-smear images (not significant), with the 20-cell gate as backup; and the moved-guardrail-timing and logging changes you specified follow.
B. **Reduce the false accepts first**, e.g. add DTD-style dot/blob textures and similar hard negatives to the not-smear class and retrain (the Phase 7 report's "no hard negatives" limitation, now showing up), then re-run this validation.
C. Keep the Phase 13 skip-for-wide-field workaround and the current guardrail.
