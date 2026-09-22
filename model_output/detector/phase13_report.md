# Phase 13 — wide-field malaria detection, Android integration

> **Phase 14 update: everything below about skipping the guardrail for wide-field malaria is historical. The guardrail was retrained on BBBC041 wide-field and single-cell images and now runs right after capture for every mode; the 20-cell gate is a secondary check. See `../guardrail/phase14b_guardrail_v3_results.md`.**

**Status: on-device session completed (Motorola Edge 50 Pro, Snapdragon 7 Gen 3, Android 16): correctness, latency and Room migration verified on the phone; see "On-device session" at the end. Nothing committed yet.**
Verified here: TFLite export and fidelity, deployed-pipeline results on 163 held-out validation fields, Kotlin/OpenCV resampling parity (unit-tested), all 39 local unit tests, debug APK + instrumented-test APK build. (Originally not verifiable here; now verified on a phone, see the end.) Photos 1-4 were not touched.

## Blockers / decisions
1. **Guardrail skipped for wide-field malaria (your decision) — BUT SEE THE NON-SMEAR TEST BELOW: the fallback is not safe on its own, decision to be revisited.** When wide-field malaria is the only condition selected, no guardrail inference runs at all; the detector's "no cells found -> Inconclusive" is the safety net. The session is logged with `guardrailResult = "skipped"` (CSV `image_check` column), counted in dashboard stats like an accepted session, and distinguishable from accepted/overridden ones. Sickle Cell and single-cell Malaria, alone or combined with anything (including wide-field malaria), run the guardrail as before. Because the choice of mode is only known after the photo is loaded, the check now runs when Analyze is tapped instead of on capture; the rejection UI (Retake / Continue anyway) and the override logging are unchanged. Original finding, kept for the record:
   **Guardrail blocks wide-field photos (found in this phase).** The bundled Phase 7 guardrail (accept if P(smear) >= 0.5) accepts **1/133 infected** and **0/30 negative** BBBC041 validation fields (median P(smear) 0.008). In the app that puts every such photo behind "Continue anyway" and excludes it from dashboard stats. It was trained before this image type existed. Fixing it means changing/retraining a model (e.g. adding BBBC041 training-split fields as smear examples and re-validating), or skipping the check in wide-field mode; both are your call. I changed nothing.
2. **Task 2 decision (unify vs keep separate paths): evidence says keep them separate. Please confirm.**
3. **Latency (Task 4): no phone number exists yet.** See below.

## Task 1 — export and verification
- **Detector -> TFLite.** Fixed 640x640 input (TFLite needs a static shape; photos are letterboxed: long side to 640, area-average when shrinking, bilinear when enlarging, pasted top-left on white), output [1,160,160,5] with the sigmoid inside. Float16 weights = **3.86 MB** (float32 = 7.65 MB). Bundled as `assets/cell_detector.tflite`.
- **Fidelity vs Keras (40 validation fields):** float16 heat-probability max |diff| 0.015, mean 5e-5; 38/40 fields with identical detection count; 99.42% of Keras detections re-found (IoU >= 0.9). Float32 is exact.
- **Deployed pipeline vs the Phase 11c Keras reference, all 163 validation fields** (TFLite detector + the *bundled* TFLite classifier, threshold 0.985): **0/163 decisions differ**, AUC 0.942 (reference 0.942), infected fields caught 120/133 = 90.2%, negative fields flagged 5/30 = 16.7% — identical to the reference for both float16 and float32 detectors. Per-field top score differs by 0.0142 on average; one field's top score differs by 0.64 without changing any decision. So float16 ships.
- **"Baseline" classifier confirmed.** The bundled `malaria_model.tflite` is byte-identical (same MD5) to `model_output/malaria_bbbc041/malaria_bbbc041_model.tflite`, the Phase 10 float32 export of `bbbc_keras.keras`, i.e. the baseline classifier that Phase 11 evaluated. The hard-negative and wider-context variants were not used.
- **Threshold confirmed from the data, not assumed:** exact in-sample threshold keeping >= 90% of infected fields is 0.98892; 0.985 sits on the same plateau (120/133 caught, 5/30 flagged). Held-out realisation from Phase 11c: 90.2% caught / 18.3% flagged (95% interval 8-35%). Shipped **t = 0.985**, detector score threshold 0.3, no box de-duplication (exactly the configuration those numbers describe).

## Task 2 — can the detector pipeline replace the single-cell path? **No (0/200).**
200 single-cell validation crops (100 infected, 100 uninfected), each treated as a whole photo:
- The detector found the true cell (IoU >= 0.5) in **0/200**; **181/200 got no detection at all**; 15 exactly one (all in the wrong place), 4 got two or more. The 12 local demo crops also give 0 detections each.
- Detect-then-classify called **0/100** infected cells positive (0/100 false positives); the existing direct path calls 97/100 positive (2/100 false positives). The paths disagree on 99/200.
- Cause: scale. A zoomed-in cell fills the frame (~550 px after letterboxing) while the detector was trained on 22-65 px cells. Not tuned around: I did not try rescaling tricks.
**Recommendation: keep two distinct paths** (as built, with a "Wide field / Single cell" choice under Malaria). If a wide-field photo yields no detected cells, the app says "Inconclusive — no cells found" and never reports "Negative" (also not logged).

## Task 3 — integration (as built)
Photo -> guardrail -> Malaria: **Wide field** (default) -> `WideFieldMalariaPipeline`: letterbox -> `CellDetector` -> 224x224 crop per cell taken from the working-resolution photo (long side capped at 2048 px to bound memory; BBBC-size fields are unaffected) -> the bundled classifier (4 threads) -> "any cell >= 0.985" -> result card. Fully offline: no network code was added.
- **Result card:** Positive/Negative, "Highest cell score", "Cells analysed: N · took X s", and the context sentence "In validation tests this check flagged about 90% of infected fields and about 18% of clean fields. Those tests used one kind of thin-smear image, so treat a result on other images with extra caution." Negative wording: "No cell was flagged — this check misses about 1 in 10 infected fields".
- The timing text is shown on the card so it can be read off the phone; it can be removed later.
- Cross-domain warning text updated (Malaria is no longer described as single-cell images only).
- No "Borderline" state for wide-field: the validated operating point is a single threshold; adding a band would be an unvalidated construct.

## Task 5 — logging (Phase 8 pattern kept)
Same session -> record structure, same result/confidence/referral fields. Added two columns to `screening_records`: `wideField` and `cellsDetected` (DB v5, **real Migration 4->5 with numeric defaults so existing logs survive**; not exercised on a device, build-verified only). Wide-field `confidence_percent` = the highest cell score; CSV gained `input_mode` (`field`/`image`) and `cells_detected` at the end. "No cells found" results are not logged. The dashboard is unchanged: malaria totals combine both modes.

## Task 4 — latency: not measured on a phone
- **Measured on this desktop (CPU, TFLite, one crop at a time, for reference only, NOT a phone):** detector 85 ms, classifier 6.2 ms per crop, mean 75.5 cells per field (max 183) -> about 0.55 s per field.
- **On a phone I have no number, and I am not going to guess one.** The app records total time and per-stage times (prepare / detect / crop / classify) in logcat and shows the total on the result card. To measure: build the debug APK, or run the instrumented test — `adb push demo_images/wide_field_eval /sdcard/Android/data/com.sicklescan.app/files/wide_field_eval`, then `./gradlew connectedDebugAndroidTest --tests "*WideFieldPipelineInstrumentedTest"` and `adb logcat -d -s WideFieldEval`. The 20-field set (12 infected + 8 negative BBBC041 validation fields, 40 MB) is at `demo_images/wide_field_eval/` (gitignored; BBBC041 is CC BY-NC-SA). The test also compares every field's decision and top score with the Python pipeline and skips itself if the folder is missing.
- **Already in place against a frozen-looking UI:** analysis runs off the main thread with live text ("Finding cells…", "Checking cells… 24 of 75").
- **If it is several seconds, options (none applied yet):** (a) keep as is with the progress text; (b) batch crops through the classifier — a modest CPU gain at best; (c) GPU/NNAPI delegate — device-dependent, needs its own validation; (d) an int8 classifier — changes the validated scores, so it needs your approval and revalidation. I recommend deciding after seeing the real number.

## Tests
39 local unit tests pass (28 existing + 11 new): resampling parity with OpenCV goldens (max 2 gray levels, mean < 0.5) for letterbox (shrink/enlarge/square) and crops (interior, shrink path, edge replication, tiny box, fractional coordinates); detection decoding; the field rule; the CSV columns. The instrumented test compiles but has not been run on a device.

## Notes
Untracked / not committed: everything in this phase. New files: `scripts/detector/*` and `scripts/classifier_v2/*` (Python), the Android sources and tests above, `assets/cell_detector.tflite`, synthetic goldens under `app/src/test/resources/goldens/`, and the reports in `model_output/detector/`. Model files in `model_output/detector/cell_detector_{float16,float32}.tflite` are archives of the exports. BBBC041-derived material (`demo_images/wide_field_eval/`) is gitignored. The BBBC041 attribution and non-commercial-terms question from Phase 12 is still open.


> **Update (Phase 14): the skip-for-wide-field workaround described below has been REMOVED. The guardrail was retrained on BBBC041 wide-field and single-cell images and now runs right after capture for every mode; the 20-cell gate stays as a secondary check. See `../guardrail/phase14b_guardrail_v3_results.md`.**

> **Update: Option A implemented (guardrail skipped + minimum-cell-count gate, k = 20). See `wide_field_results.md` for the shipped behaviour and the stated limitation; the gate check is `phase13_gate_check.txt`.**

## Non-smear safety test: is "no cells found -> Inconclusive" a safe net without the guardrail? **No.**
Deployed pipeline (float16 detector, decode 0.3, crops, bundled classifier, t = 0.985) on 60 images per category from the guardrail's own negative data (fixed seed; images stay local). What the app would show:

| Category | n | Inconclusive (safe) | Negative | **Positive** | median cells found |
|---|---|---|---|---|---|
| Everyday objects (COCO) | 60 | 3 | 50 | **7 (11.7%)** | 8 |
| Textures (DTD) | 60 | 17 | 39 | **4 (6.7%)** | 2 |
| Scenes (Places) | 60 | 2 | 57 | **1 (1.7%)** | 9 |
| Faces (LFW) | 60 | 7 | 53 | 0 | 2 |
| Screenshots | 60 | 17 | 43 | 0 | 2 |
| Documents | 60 | 49 | 11 | 0 | 0 |
| Synthetic capture failures: blank, black, blur, finger (240 total) | 240 | 240 | 0 | 0 | 0 |
| **Natural images total** | **360** | **95 (26%)** | **253 (70%)** | **12 (3.3%)** | 3 |

- Blank / black / blurred / finger-over-lens photos are all safely Inconclusive.
- **Real-world non-smear photos are not**: the detector finds spurious "cells" (median 3, up to 162) in most of them, the classifier scores them, and **74% end in a verdict**: 70% would say **Negative** ("No cell was flagged...", e.g. on a selfie or a screenshot: misleading) and **3.3% Positive** (a false referral; 11.7% for everyday-object photos, with top scores up to 1.0).
- So skipping the guardrail is **not** safe with "no cells found" as the only safeguard.

### Candidate fix measured (analysis only, not built): minimum-cell-count gate
Real BBBC041 fields are dense (cells per field: min 19, 5th percentile 28, median 69); natural non-smear images are sparse (median 3, 95th percentile 26). A rule "fewer than k detected cells -> Inconclusive":

| k | BBBC infected fields caught | BBBC negatives flagged | non-smear images still given a verdict | non-smear images shown Positive |
|---|---|---|---|---|
| 1 (current) | 90.2% | 16.7% | 73.6% | 3.3% |
| 10 | 90.2% | 16.7% | 24.7% | 2.8% |
| **20** | **90.2%** | **16.7%** | **8.1% (29/360)** | **1.1% (4/360)** |
| 30 | 85.7% | 16.7% | 4.2% | 0.8% |
| 40 | 80.5% | 13.3% | 1.7% | 0.6% |
k = 20 leaves every validated BBBC decision unchanged (the sparsest validation field has 19 cells, which is not a flagged one) and removes ~90% of non-smear verdicts, but: it is chosen in-sample on 163 fields from one microscope and dense monolayers; sparser real fields (other stains, phone photos, low-density smears) would become Inconclusive; and 8% of non-smear photos (coco 7, dtd 11, places 6, screenshots 5 of 60) still get a verdict.

### Options (your call)
A. Skip the guardrail **and add a minimum-cell-count gate** (k about 20): quick, no retraining; validate the count on more thin smears (esp. sparse ones) before trusting it; residual ~1% false Positive on non-smear photos.
B. **Retrain the guardrail with BBBC041 fields as smear examples** and use it in wide-field mode too (the current one rejects 99.4% of them): the designed mechanism, needs a model change and validation on these non-smear images; still unvalidated on phone photos of other stains.
C. Both A and B.
D. Keep as is: not recommended (a selfie can get a Negative verdict and an everyday photo can get a Positive one).
Raw output: `phase13_nonsmear_test.txt`, `phase13_nonsmear_per_image.csv`, `phase13_cellcount_gate.txt`; scripts `nonsmear_test.py`, `cellcount_gate_analysis.py`.


## On-device session (Motorola Edge 50 Pro, Snapdragon 7 Gen 3 / SM7550, 8 cores, Android 16) — real numbers
Test set: 20 BBBC041 validation fields (12 infected + 8 negative); reserved Photos 1-4 not used. Raw logs: `phase13_device_run1_baseline.txt`, `phase13_device_benchmark.txt`, `phase13_device_run3_final.txt`.

### (a) Correctness on the phone vs the Python pipeline
Kotlin + TFLite on the phone vs the Python TFLite pipeline: **0/20 answers differ** (answer = Positive / Negative / Inconclusive, including the 20-cell count gate), **0/20 top-cell scores differ by more than 0.05**, **detected cell counts identical on all 20 fields**. (The largest top-score difference is 0.034, on one 183-cell negative field: 0.363 on the phone vs 0.397 in Python; every other field is within 0.02, and every answer is the same.) The instrumented test passes.
The count gate itself was not exercised by this set (no field has fewer than 20 cells); it is covered by unit tests, and by the phone's real usage log below.

### (a) Latency, per field (wall clock on the phone)
| | Median | Mean | Max (183-cell field) | First analysis |
|---|---|---|---|---|
| Original build (one crop at a time, crops on one thread) | 3.47 s | 3.66 s | 7.71 s | 7.3 s |
| **Final build (shipped defaults)** | **2.77 s** | **2.89 s** | **5.81 s** | **4.0 s** (models warmed up when wide-field is selected, ahead of Analyze) |
Final mean per stage: prepare 144 ms, detect 316 ms, crop 439 ms, classify 1,989 ms (about 26 ms per cell). Fields have 37-183 cells (median about 75). Timings vary with the phone's temperature: a later run right after several long back-to-back runs was slower in the classifier stage (about +10%).

### (b) Crop batching
Per your rule (several seconds, so implement batching first). Controlled benchmark, 10 fields, every setting gave **identical top scores (max difference 0.000000)**:
- Batching the classifier calls alone (batch 8, 16): **no speed-up** (classify 1,923 ms -> 1,950 ms): on this CPU the classifier is compute-bound, roughly 25 ms per crop however it is grouped.
- **The gain came from building the crops in parallel** (batch of 8 crops filled by 4-6 threads): crop stage 1,281 ms -> 435-486 ms, mean total 3.67 s -> 2.85-2.92 s. This went slightly beyond the literal "batching" you approved, because batching is what makes the parallel crop building possible; it is also score-neutral (verified above). 6 classifier threads instead of 4: no change.
- Warm-up: running the crop path and models once in the background as soon as wide-field malaria is selected removed most of the first-use penalty (first analysis 7.3 s -> 4.0 s).
- **Not done, needs your approval:** anything that changes numerics: int8, fp16 compute, GPU delegate. The classifier stage (about 2 s of about 2.9 s) is the remaining cost.

### (c) Room migration 4 -> 5 on the phone with the real existing data
Installed the new build over the existing debuggable app (`install -r`, no uninstall) after backing up its database. Before: user_version 4, **30 sessions, 32 records, 19 guardrail events**. After opening the app: **user_version 5, `wideField` and `cellsDetected` present as INTEGER NOT NULL DEFAULT 0, all rows identical on every original column (checked row by row), new columns 0 on all 32 old rows, foreign_key_check empty, integrity_check ok, no exception in logcat, Dashboard displays the data.**
Later database check after the app had been used on the phone (not by my test runs): the original 30 / 19 / 32 rows still unchanged, and 6 new sessions / 6 new records logged correctly, including three wide-field checks logged as `result = inconclusive`, `wideField = 1`, `cellsDetected` 4 / 2 / 1 in sessions with image check `skipped` (the count gate on real photos), two overridden and one accepted session from the guardrail flow.
Scripts: `scripts/device/inspect_db.py`, `compare_dbs.py`; backups are local only (`demo_images/device_checks/`, gitignored; they contain real usage data).

### Cleanup on the phone
Removed the test APK (`com.sicklescan.app.test`) and the pushed evaluation images; the phone keeps the updated app with all data.
