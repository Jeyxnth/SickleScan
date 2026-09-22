# Phase 11b — Step 0 and Step 1 results (approved plan; step 2 not built)

**Bottom line: the field-context hypothesis is not supported, and hard negatives change the operating point, not the discrimination.** Neither cheap experiment beats the baseline crop classifier at telling infected fields from clean fields. This is evidence against building the joint model *for accuracy*; the remaining case for it is on-device efficiency (one forward pass instead of ~73 crop classifications per photo).

Setup (all comparisons identical): BBBC041 held-out validation split, 133 infected fields vs 30 malaria-negative fields (no infected / difficult box), the same cached detector boxes (detector score >= 0.3), same recipe/seed/augmentation for every classifier; only the training data differs. Image-level score = max classifier score over the field's detected cells.

## Step 0 — the 41 flagged cells (non-expert visual read; montage `step0_flagged_cells.png`, local)
Roughly 15 are not red blood cells at all (large nucleated / leukocyte-like cells, dark pigment or debris clumps — one clump was hit by three overlapping detections — and fragments at the image edge); roughly 20 are ordinary-looking RBCs with small dark dots or stipple that could be unannotated ring-stage parasites or platelets/debris; a handful show nothing visible. So some "false positives" may be label noise, and part of the measured rate is not fixable by any model. I cannot adjudicate the dot-bearing cells; a parasitologist review of that montage would settle it.
Also observed: duplicate detections inflate count-based rules; box de-duplication (NMS 0.5) changed the numbers only slightly (per-cell 1.86% -> 1.74%, no change in fields flagged).

## Step 1 — results
| Classifier | Image AUC (95% CI) | At the app's 0.5 boundary: infected fields caught | negative fields flagged | per-cell flag rate on negatives |
|---|---|---|---|---|
| Baseline (crop x1.15) | 0.942 (0.898-0.976) | 131/133 = 98% | 15/30 = 50% | 1.86% |
| **1b Wider context** (crop x2.0) | 0.936 (0.890-0.973) | 129/133 = 97% | 18/30 = 60% | 1.62% |
| **1a Hard negatives** (+4,846 mined) | 0.934 (0.890-0.970) | 116/133 = 87% | 3/30 = 10% | 0.24% |

Paired bootstrap over fields: AUC(hard negatives) - AUC(baseline) = -0.008 [-0.031, +0.015]; AUC(wide) - AUC(baseline) = -0.006 [-0.037, +0.025]. **No detectable difference in either direction.**

**Hard negatives moved the operating point along the same curve.** The baseline at higher thresholds gives the same trade-off:
| Baseline threshold | infected fields caught | negative fields flagged |
|---|---|---|
| 0.5 | 98.5% | 50% |
| 0.9 | 93.2% | 30% |
| 0.985 | 90.2% | 17% |
| 0.995 | 88.0% | 13% |
| 0.999 | 75.2% | 3% |
| *hard-negative model @0.5* | *87.2%* | *10%* |
Hard-negative training effectively lowered the prior of "infected" (the original classifier was trained at ~23% infected crops, versus ~3% in real fields), which is what a threshold increase does. The wide-context model shows no gain (60% flagged at 0.5 is one field per 3.3 points, i.e. within noise of the baseline's 50%).

**Against the pre-agreed target (>= 90% of infected fields caught with <= 15% of negative fields flagged):** met only at in-sample thresholds (baseline 90.2%/17% at 0.985 is just outside; 1a reaches 90%/13% at t=0.403), but with 30 negative fields one field is 3.3 points and the 95% interval on 13% is roughly 5-30%. It cannot be claimed as validated, and picking the threshold on these same fields is optimistic.

## What this means
1. **Field context in the crop is not the missing ingredient** (wider crops: no gain). This tests context at the classifier's input; it is not a test of a joint model's learned features, but it is the cheapest evidence available, and it points against.
2. **The image-level AUC of ~0.94 looks like a ceiling shared by every variant here.** Possible causes, none confirmed: label noise / unannotated parasites (Step 0), true look-alikes (WBC-like cells, debris), or the intrinsic difficulty of BBBC041. A new architecture is unlikely to move it without new information.
3. **The false-alarm rate is a threshold choice, not a model defect**, on this evidence: the app can trade sensitivity for false alarms along the curve above.
4. **Step 2 (joint model):** not recommended for accuracy reasons. The remaining argument is efficiency and packaging (single model, one pass). That is your call.

## Limits of this evidence
Only 30 negative fields (3 in the official test split), one training seed per variant, hard negatives were mined on training images where the detector/classifier err less than on unseen ones, detector threshold fixed at 0.3, P. vivax thin smears from a single stain protocol, and "negative" means annotators labelled no infected cell. Nothing here addresses the score compression seen on the shifted test split or thick-smear photos.

Scripts: `scripts/classifier_v2/{mine_hard_negatives,build_wide_crops,train_variant,eval_variants,bootstrap_compare}.py`. Raw outputs: `step1_eval.txt`, `step1_bootstrap.txt`. Models: `artifacts/classifier_v2/` (local). Nothing committed; the shipped Android model is unchanged.
