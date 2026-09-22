# Guardrail v3 results (Phase 14b) — the guardrail now bundled in the app

The bundled `guardrail_model.tflite` is **v3**: the Phase 7 recipe (MobileNetV2, dense128 head, same seed, augmentation, early stopping and class weights), trained with BBBC041 **wide-field fields** (822 training-split images) and BBBC041 **single-cell crops** (800, training split) added to the smear class. The 181 held-out validation images (which contain the 163 fields used in Phases 11-13), the 120 official test images and the held-out single-cell crops were never used in training (the field split is enforced by assertion; crops are assigned by their source image's split). Threshold unchanged: accept when P(smear) >= 0.5. Float16 TFLite, 4.79 MB; 0/2,777 decisions differ from Keras (max probability difference 0.048). Scripts: `scripts/guardrail_v2/`; raw output: `phase14b_eval_raw.txt`.

## Full validation, identical images for all three models (v1 = Phase 7, v2 = wide-field added, v3 = wide-field + single-cell added)
| Set | Want | v1 | v2 | **v3 (shipped)** |
|---|---|---|---|---|
| 163 held-out BBBC041 wide-field fields | accept | 1/163 (0.6%) | 163/163 | **163/163 (100%)** |
| 120 official BBBC041 test images (different microscope) | accept | 56/120 (46.7%) | 118/120 (98.3%) | **110/120 (91.7%)¹** |
| Sickle-cell test images (86, held out) | accept | 86/86 | 86/86 | **86/86** |
| NIH single-cell malaria test images (387, held out) | accept | 387/387 | 387/387 | **387/387** |
| BBBC041 single-cell validation crops (1,823, held-out images) | accept | 144 (7.9%) | 153 (8.4%) | **1,822/1,823 (99.9%)** |
| BBBC041 single-cell official-test crops (5,917, different microscope) | accept | 378 (6.4%) | 222 (3.8%) | **2,508/5,917 (42.4%)** |
| Phase 13 non-smear sample: 360 natural + 240 synthetic (70% of it was in the guardrail's training split, so weak evidence) | reject | 600/600 | 600/600 | **600/600** |
| Held-out real non-smear (945) | reject | 944 | 943 | **943/945 (99.8%)** |
| FRESH never-used non-smear (986) | reject | 985 | 982 | **985/986 (99.9%)** |
| **False accepts on all 1,931 unseen non-smear** | | 2 (0.10%) | 6 (0.31%) | **3 (0.16%)** |

¹ Measured on-device (real phone, not desktop) on these same 120 images. The figure computed with the Python/TensorFlow preprocessing used for the rest of this table (and for the 95% interval above) is 111/120 (92.5%) [86.4-96.0%]; the difference is one boundary-adjacent image where Android's JPEG decoder produces slightly different pixels than TensorFlow's, not a resize or model issue — see `../detector/phase14c_preprocessing_root_cause.md`. Every other row in this table showed 0 flips on-device.

## Was the change worth it? Yes, with two things to know
- **Fixed:** wide-field photos are no longer rejected (0.6% -> 100% on the held-out fields; 46.7% -> 92.5% on a different microscope), and BBBC041-style single-cell crops are accepted (8.4% -> 99.9%). Sickle-cell and NIH single-cell images are unaffected. Non-smear rejection on unseen images is at least as good as the wide-field-only model (0.31% -> 0.16% false accepts).
- **Cost 1 - shifted microscopes:** the 120 official test images (a different microscope) dropped from 98.3% (v2) to 92.5% accepted as computed with Python/TensorFlow preprocessing (91.7% as measured on the phone¹), i.e. about 7.5-8.3% of wide-field photos from a different setup would get the "doesn't look like a smear" soft warning (Retake / Continue anyway; Continue anyway saves the result but excludes it from the dashboard stats). And **single-cell crops from that different microscope are still mostly rejected (57.6%)**, because the guardrail's single-cell examples come from one setup. Cross-microscope robustness of the guardrail remains limited.

## ACCEPTED RESIDUAL RISK (stated plainly)
The guardrail now lets through a small, statistically inconclusive rate of texture / pattern non-smear images: **about 0.2-0.3% (3/1,931 = 0.16% for the shipped v3; 6/1,931 = 0.31% for the wide-field-only v2; the difference from the original model, 2/1,931 = 0.10%, is not statistically significant, Fisher exact p = 0.29 for v2 vs v1)**. Every image accepted so far is a dot / blob pattern from the DTD textures set (the same kind of pattern each time, though the individual images differed between retrains, which shows the estimate is noisy). The risk is accepted because:
1. **None of the images accepted so far produced a false Positive downstream** (v2's six and v3's three, run through the detector, classifier and cell gate): 4 of the 9 were stopped by the 20-cell count gate (Inconclusive), 5 would display **Negative** (a misleading but not alarming verdict), 0 displayed Positive.
2. **The 20-cell count gate remains as a secondary check** for wide-field malaria (it cost nothing to keep).
Not covered: real phone photos through a microscope attachment, other dot-pattern non-smear images we have not sampled, and the possibility that a Negative verdict on a non-smear photo misleads a user. These numbers are estimates on public-dataset photos, from one training run per model.

## Other checks
- Python vs phone (Motorola Edge 50 Pro, Snapdragon 7 Gen 3): on-device guardrail accepts 20/20 of the wide-field validation fields (min P(smear) 0.875) and 12/12 BBBC041 single-cell crops (min 0.999); the on-device sample-image test (8 must-accept, 6 must-reject) passes.
- The old (Phase 7) results and limitations are in `guardrail_results.md`; its "no hard negatives" limitation still applies (no other microscopy or histology images were included as negatives).
