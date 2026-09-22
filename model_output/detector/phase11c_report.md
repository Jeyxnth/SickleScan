# Phase 11c — learned field-level aggregation vs hand-picked rules

**Result: the learned aggregator does not beat the simplest hand-picked rule, "any cell >= t".** On held-out folds it is no better at any operating point (differences of 0-3 points against a sampling uncertainty of about +-13 points) and its threshold-free discrimination is lower (AUC 0.916 vs 0.942). The count-based rules used in earlier phases (">= 2 / >= 3 cells >= 0.5") are the worst option. No aggregator is worth shipping on this evidence.

## Setup
- Per-cell scores from the **baseline** classifier (the currently shipped one) on the same 133 infected + 30 negative BBBC041 validation fields, detector score >= 0.3. Chosen because it has the highest image-level AUC (0.942) — though all three variants were within noise (0.934-0.942), so this is not a selection among real differences. The hard-negative classifier was run as a robustness check and gives the same conclusions (`phase11c_aggregator_hardneg.txt`).
- Field features: max score, mean of top-3, mean of top-5, count >= 0.5 / 0.7 / 0.9, number of cells (scores logit-transformed, counts log1p).
- Models: L2 logistic regression, class-balanced, regularisation picked by inner 3-fold CV (nested), plus a 2-feature version (max score + number of cells) as a lower-capacity check.
- **Protocol (nothing tuned on held-out fields):** repeated stratified 5-fold CV, 20 repeats. For *every* method — learned and hand-picked — the operating threshold is set on the training folds only (largest threshold reaching the target sensitivity there), then applied to the held-out fold. Numbers below are the realised results on held-out fields. The hand-picked rules therefore get the same treatment (their earlier "best threshold" numbers were chosen in-sample and were optimistic).

## Threshold-free discrimination (image AUC, held-out)
| Method | AUC |
|---|---|
| "any cell >= t" (max score; parameter-free) | **0.942** |
| ">= 2 cells >= t" | 0.826 |
| ">= 3 cells >= t" | 0.747 |
| learned LR, 7 features (CV, 20 repeats) | 0.916 (range 0.891-0.930) |
| learned LR, 2 features | 0.915 |
The learned models score *lower* than the plain maximum, which is the overfitting signature expected at this sample size (30 negatives).

## Realised operating points (held-out; mean of 20 repeats; Wilson 95% interval for the negative-field rate, n = 30)
| Target sens. (train folds) | Method | Infected fields caught | Negative fields flagged |
|---|---|---|---|
| 85% | any cell >= t | 85.0% | 10.0% (3-26%) |
| 85% | learned LR, 7 feat | 84.5% | 11.5% (4-27%) |
| **90%** | **any cell >= t** | **90.2%** | **18.3% (8-35%)** |
| 90% | learned LR, 7 feat | 89.2% | 18.7% (9-36%) |
| 90% | learned LR, 2 feat | 89.8% | 16.8% (7-34%) |
| 90% | >= 2 cells >= t | 89.7% | 60.0% (42-75%) |
| 90% | >= 3 cells >= t | 89.5% | 76.5% (59-88%) |
| 95% | any cell >= t | 94.5% | 36.2% (21-54%) |
| 95% | learned LR, 7 feat | 93.5% | 33.5% (19-51%) |
The learned aggregator and "any cell >= t" are indistinguishable (the 2-3 point differences are far inside the intervals, and they go in both directions). Same picture for the hard-negative classifier (e.g. 90% target: 19.0% vs 17.5%).

## Against the best hand-picked rules from Step 1 / 11a
- **"any cell >= 0.5"** (the app's current boundary): 98.5% caught, 50% of negative fields flagged. The learned aggregator does not fix this; the only way down is raising the threshold on the max score (the aggregator learns essentially that: its largest coefficient is on logit(max)).
- **">= 3 cells >= 0.5"** (58.6% caught / 20% flagged) looked attractive as a fixed point but is **dominated** by a higher threshold on the max score (e.g. baseline at 0.999: 75% caught / 3% flagged, from Step 1), and its full curve is the worst of all (76% flagged at 90% sensitivity). Count-based rules do not help because false alarms in negative fields are as likely to be multi-cell (clumps, duplicates, debris) as single-cell.
- **Honest replacement for the earlier in-sample figure:** the simple max rule gives about **90% caught with ~18% of negative fields flagged (8-35%)**, not the 13-17% read off the curve earlier. The pre-agreed target (>= 90% caught with <= 15% flagged) is **not met** out-of-sample, and the uncertainty band straddles it.

## Caveats
30 negative fields (a single field is 3.3 points); CV repeats are re-splits of the same 163 fields, not new data, so the "sd across splits" understates true uncertainty (the Wilson intervals are the honest ones); features use the detector's cells at score >= 0.3 without box de-duplication; thin-smear P. vivax from one stain protocol. Important for shipping: any threshold set on these fields (e.g. ~0.985 for 90% sensitivity) is calibrated to this microscope setup; Phase 9b showed classifier scores compress under a different setup, so a fixed threshold will not transfer and would need per-setup validation.

Scripts: `scripts/classifier_v2/learned_aggregator.py`. Raw output: `phase11c_aggregator_baseline.txt`, `phase11c_aggregator_hardneg.txt`. Nothing committed; shipped Android model unchanged.
