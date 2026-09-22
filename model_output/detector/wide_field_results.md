# Wide-field malaria detection — results (what ships, what was measured, what is NOT validated)

Pipeline: photo -> CenterNet cell detector (float16 TFLite, 640x640 letterbox, detector score >= 0.3) -> 224x224 crop per detected cell -> BBBC041 malaria classifier (baseline, float32) -> field answer:

| Detected cells | Answer |
|---|---|
| fewer than 20 (including none) | **Inconclusive** (count gate) |
| 20 or more, any cell scores >= 0.985 | **Positive** |
| 20 or more, no cell >= 0.985 | **Negative** |

The image check (guardrail) runs right after capture for this mode too (retrained in Phase 14; the Phase 13 skip-for-wide-field workaround is removed), and the session is logged accepted / overridden like every other mode; the 20-cell gate is the secondary check. Full details: `phase13_report.md`; earlier evidence: `phase11a_report.md`, `phase11b_step1_report.md`, `phase11c_report.md`.

## Measured performance (held-out BBBC041 validation fields: 133 infected, 30 negative; thin smears, P. vivax, Giemsa, one microscope setup)
- Deployed TFLite pipeline (float16 detector + bundled classifier), threshold 0.985: **infected fields caught 120/133 = 90.2%; negative fields flagged 5/30 = 16.7%**; image AUC 0.942; 0/163 decisions differ from the Keras reference.
- Out-of-sample (threshold chosen on training folds): about 90% caught / **18% flagged (95% interval 8-35%)** — only 30 negative fields, so the false-alarm rate is imprecise.
- **Count gate (k = 20): 0/163 Positive/not-Positive decisions change** (still 120/133 and 5/30). **One** answer changes: a negative validation field with 19 detected cells goes from Negative to Inconclusive. The sparsest infected validation field has 22 cells (validation cells per field: min 19, 5th percentile 28, median 69).

## Why the count gate exists
(Written when the guardrail was skipped for this mode in Phase 13; the gate is now a secondary check behind the retrained guardrail.) With no guardrail, "no cells found -> Inconclusive" alone was not a safe net: on 360 natural non-smear photos (objects, textures, scenes, faces, screenshots, documents) the detector found spurious cells, and 74% ended in a verdict (70% Negative, 3.3% Positive; 11.7% Positive for everyday-object photos). Blank, black, blurred and finger-over-lens photos were all safely Inconclusive. With the gate at 20, 8.1% of the 360 still get a verdict and **1.1% (4/360) get a false Positive**. This is reduced risk, not eliminated risk.

## KNOWN LIMITATION — stated exactly, NOT a solved problem
**k = 20 was tuned on dense lab-microscope BBBC041 fields. It has NOT been validated against sparser real phone-captured smears, where a genuinely infected but sparse field could incorrectly fall below the gate and return Inconclusive instead of a real result.**
Related, also unvalidated: the 0.985 threshold and the ~90% / ~18% figures are for BBBC041-style images only, and classifier scores compress on unfamiliar microscopes and cameras; the non-smear numbers come from one 360-image sample from the guardrail's own negative data; and the gate was chosen on the same 163 fields it was checked against (in-sample).
Consequences for use: an Inconclusive on a real field photo may mean "too sparse or not a dense thin smear" (the app says so on screen) and is not evidence of absence or presence. Sparse but genuine smears need their own validation (real phone-captured thin smears with lab-confirmed status) before the gate value can be trusted or tuned.

## Logging (visible in the data)
Every wide-field check is logged as a Room record with `wideField = true` and `cellsDetected`; a count-gate Inconclusive is logged as `result = "inconclusive"` (confidence column = highest cell score, referral not flagged) in a session whose image check is `accepted` / `overridden` (Phase 13 sessions logged before the guardrail was retrained keep the legacy value `skipped`). In the CSV: `input_mode = field`, `cells_detected = N`. Inconclusive records are excluded from the dashboard's positive/negative statistics and reported on the dashboard as a separate count.

## Verified on a phone (Motorola Edge 50 Pro, Snapdragon 7 Gen 3), 20 BBBC041 validation fields
- Same answers as the Python pipeline on all 20 fields (0 differences including the count gate), identical cell counts, no top score differing by more than 0.05.
- **Latency per field: median 2.77 s, mean 2.89 s, max 5.81 s** (37-183 cells); first analysis 4.0 s after a background warm-up. Crop building is parallel; the classifier (about 26 ms per cell, about 2 s per field) is the remaining cost, and reducing it further would change numerics (int8 / fp16 / GPU) and needs approval and re-validation.
- Room migration 4 -> 5 completed cleanly on the real existing data (30 sessions, 32 records, 19 guardrail events unchanged, new columns default 0, integrity ok).
- Not verified: the count gate on sparse real phone-captured smears (see the limitation above); behaviour on other phones (timings are for this one).

## Accepted residual risk of the retrained guardrail (Phase 14b) — stated plainly
The guardrail that now runs before wide-field malaria lets through a small, statistically inconclusive rate of texture / pattern non-smear images: **about 0.2-0.3% (3/1,931 = 0.16% for the shipped model; 6/1,931 = 0.31% for the wide-field-only version; the original model was 2/1,931 = 0.10%; no statistically significant difference)**, all dot / blob patterns from a textures dataset. It is accepted because none of the images accepted so far produced a false Positive downstream (4 of 9 were stopped by the 20-cell gate, 5 would show a misleading Negative, 0 a Positive), and because the 20-cell count gate remains as a secondary check. The guardrail also rejects about 7.5-8.3% of wide-field photos from a different microscope (92.5% accepted per Python/TensorFlow preprocessing, 91.7% as measured on-device — the difference is one boundary-adjacent image where Android's JPEG decoder differs slightly from TensorFlow's, not a resize or model issue) and most single-cell crops from that microscope (42.4% accepted); rejected photos get a soft warning and can be continued. Details: `../guardrail/phase14b_guardrail_v3_results.md`.
