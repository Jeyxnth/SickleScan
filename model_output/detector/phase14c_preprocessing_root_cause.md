# Root cause of the SharedImageBothDiseasesInstrumentedTest discrepancy (sickle model on a malaria cell image: 0.852 on the phone vs 0.806 expected)

## Answer in one paragraph
The model file is untouched and the phone's runtime is bit-for-bit consistent with the desktop; the difference is entirely the **resize step of the app's preprocessing**. Android's `Bitmap.createScaledBitmap(filter=true)`, which TensorFlow Lite Support's `ResizeOp(BILINEAR)` uses, is an 8-bit fixed-point bilinear (sub-pixel weights quantised to 1/16 of a pixel, result truncated to a whole 0-255 value), whereas the Phase 8 expected value was computed with TensorFlow's float bilinear resize (`tf.image.resize`), the same preprocessing the models were trained and validated with. The two pipelines feed the model slightly different pixels (mean 0.46 gray levels, up to 18.5 at edges, on this image), and the sickle model, which is out of its domain on a malaria cell and therefore steep there, turns that into 0.045 in probability. This is not new drift: the app's path has been this way since Phase 2, and the test's expected value was never a value the app could produce on Android.

## What was checked
1. **Model file:** `sicklescan_model.tflite` in the app is byte-identical to git HEAD (MD5 `33a6b2c912078166d72693c7313496c3`); only the Phase 1 and Phase 2 commits ever touched it; `git status` shows it unmodified.
2. **Did fixing the wrong-context bug change the preprocessing?** No. The fix only changed which `AssetManager` supplies the test image's bytes (the test APK's, where the images live, instead of the app's, which does not have them). The bytes are the same file, and the app's own code path (`ImageClassifier`: `TensorImage` -> `ResizeOp(224,224,BILINEAR)` -> `NormalizeOp(127.5,127.5)`) has not changed since Phase 2. The test was failing to open its images before, which is why it never produced a number.
3. **Exact preprocessing paths compared.**
   - Original expected value: `_pair_probs.py` -> `scripts/guardrail/data_pipeline.load_and_preprocess_single`: `tf.io.decode_image` -> `tf.image.resize(..., [224,224])` (float bilinear, half-pixel centres, no rounding) -> `(x-127.5)/127.5` -> desktop TFLite interpreter.
   - App: `BitmapFactory` decode -> `createScaledBitmap` bilinear (8-bit) -> normalise -> TFLite on the phone.
4. **On-device decomposition (Motorola Edge 50 Pro), target image `parasitized_1.png`:**
   - decoded pixels: **identical** to Python's (max difference 0; all 8 PNGs identical; the 17 JPEGs differ by up to 3-9 gray levels, a JPEG-decoder difference unrelated to this image);
   - final tensor: different (mean 0.46, 99th percentile 6.6, max 18.5 gray levels; phone values are whole numbers and on average 0.36 darker; the differences follow image edges, correlation 0.72, and are not a spatial shift);
   - **same phone, same model, fed Python's tensor: P = 0.806090 (equals the Phase 8 value); fed the app's own tensor: P = 0.851534.** So the model and the phone runtime agree exactly with the desktop; only the input tensor differs.
5. **Mechanism confirmed by emulation:** half-pixel-centre bilinear with sub-pixel coordinates/weights quantised to 1/16 and truncation to 8 bit reproduces the phone's tensor with mean error 0.006 gray levels over all 25 images (0.001 and 99.9% exact pixels on the target image; up- and down-scaling, PNG and JPEG); every variant without the quantisation/truncation is much worse (`phase14c_resize_emulation_validation.txt`). Fed through the model, the emulation predicts the phone's sickle probability to within **0.0015** on all 25 images (0.852 vs 0.8515 on the target), whereas TensorFlow-style preprocessing is off by up to 0.065 (`phase14c_emulation_vs_phone_probs.txt`).

## Does it matter beyond this one out-of-domain image? Measured over whole populations (`phase14c_preprocessing_impact.txt`)
- **Sickle model on all 569 sickle images:** mean |dP| 0.0024, max 0.050, only 5 images above 0.03, **0 decision flips at 0.5**, 2/569 change the app's Positive/Borderline/Negative status (65% band), agreement with labels identical (0.9367 both ways; these images were also used to train the model, so this is parity, not held-out accuracy).
- **Guardrail v3 (Phase 14 validation was done with TensorFlow-style preprocessing):** **0 decision flips** on the 163 BBBC fields, 120 BBBC test images, 86 sickle, 387 NIH, 945 held-out non-smear and 986 fresh non-smear images (largest single probability shift 0.11, on a 120-set image that stays above 0.5; single-cell crops are already 224x224 so no resize happens). The Phase 14 numbers therefore hold under the app's real preprocessing (JPEG-decoder differences, a second small effect, were not included in this population measurement).
- The 25 images on the phone: sickle max |dP| 0.065 (a malaria cell, out of domain), malaria model max 0.029, guardrail max 0.022; no decision flips.
- Why this one test image looks worse: the sickle model on a malaria cell returns a mid-range, steep-response value (0.41-0.85 on these crops), so the same small input change moves it more than it moves in-domain inputs.

## Which number is more trustworthy?
Neither is wrong; they answer different questions.
- **0.806** is what the model computes under the preprocessing it was trained and validated with (all documented accuracy numbers use it). Trust it as the model's reference behaviour.
- **0.852** is what the app actually outputs on Android. Trust it as what a user will see, and as the correct expectation for a device test. 0.806 is a value the Android app cannot produce.
The Phase 8 test comment ("identical preprocessing") was wrong about the resize step; it was a Python-vs-Android parity claim that had never been checked on a device.

## Options
1. **Fix the test (recommended):** pin per-row expected values computed with the phone's preprocessing (validated emulation, or the values measured on the phone: `parasitized_1` sickle 0.8515, `pos_20` sickle 0.9686) with a tight tolerance (0.01), and say in the comment that they are Android-resize values. Do not just widen the tolerance to 0.07: that would weaken the in-domain rows, which currently pass with the old numbers.
2. **Optionally fix the app instead:** make the disease classifiers' resize float-exact like TensorFlow (as the wide-field pipeline already does with `ImageOps`), so the validated numbers apply bit-for-bit. Measured benefit is small (0 flips at 0.5 in the populations above, 2/569 sickle status changes), so I would not do it unless you want exact validation parity.
3. Leave the test failing (not recommended).

---
# Resolution (Phase 14d): the classifiers now use a TensorFlow-exact resize
`ImageOps.resizeBilinearTf` / `classifierInput` (float, half-pixel centres, two taps even when shrinking, no rounding, source read row by row) replaces `createScaledBitmap` in `ImageClassifier`, so the sickle-cell, single-cell malaria AND guardrail classifiers (they share that class) now match `tf.image.resize`, the call used to train and validate them. The OpenCV-exact resize the wide-field pipeline uses was NOT reused for this: it area-averages when shrinking, which TensorFlow does not do, and would have created a new discrepancy on downscaled sickle photos. The wide-field pipeline is unchanged.
Verification (exact Kotlin code on the JVM, TensorFlow-decoded pixels, real bundled models; `phase14d_kotlin_resize_verification.txt`): max |dP| 0.000014 over all 25 test assets, all 569 sickle images, 387 NIH crops, 200 BBBC crops, 163 BBBC fields, 120 BBBC test images, 945 held-out and 986 fresh non-smear images; 0 decision flips, 0 app-status changes; `parasitized_1` on the sickle model 0.8061 (TensorFlow) = 0.8061 (Kotlin), old phone path 0.8515. Expected on the phone (from phone-decoded pixels, `phase14d_decoder_residual_estimate.txt`): PNGs exact, JPEGs keep a small decoder-only residual (sickle mean 0.005, max 0.03; guardrail max 0.011; no flips). On-device confirmation with the real BitmapFactory path: pending (phone not connected); runner `ParityRunInstrumentedTest` + `scripts/device/parity_device.py build|analyze`.

---
# Phase 14d, final on-device measurement — real, small, precisely-quantified changes to two already-reported numbers

**Everything below was measured on the real phone (Motorola Edge 50 Pro) running the actual app code (BitmapFactory decode + the new `ImageOps.resizeBilinearTf` + the real bundled TFLite models), not estimated.** All 4,032 parity jobs ran on-device; see `phase14d_device_parity_results.txt`, `phase14d_flip_identification.txt`, `phase14d_test_split_impact.txt`.

## A mistake in my own verification harness, found and corrected
The first on-device run of the 569 sickle images showed 119 "decision flips" — far more than expected. Root cause: **`Positive/Unlabelled` and `Negative/Clear` share 147 identical filenames** (both numbered `1.jpg`, `2.jpg`, ...). My test harness (`scripts/device/parity_device.py`) copied both to the same `sickle/<basename>` path on the phone, so the second copy silently overwrote the first for 147 of 569 images — the phone was scoring the wrong image for those files. This is a bug in my verification script, not in the app or the resize fix. Fixed by keeping the source folder in the destination name (`sickle/positive__<name>` / `sickle/negative__<name>`); re-verified no collisions anywhere in the full job set; re-ran on-device. Flagging this so the numbers below are trusted for the right reason, not taken on faith.

## Real result after the fix: 3 genuine flips out of 569 (not 119)
All three sit almost exactly on the 0.5 boundary (TensorFlow reference 0.49–0.52) and are the expected residual: Android's JPEG decoder differs slightly from TensorFlow's (the resize itself is proven exact to 1e-5, see above), and on an image whose true output is that close to the threshold, that residual is enough to flip it. This is unrelated to and not fixable by the resize change.

## Does it change already-reported numbers? Yes, twice, both small
**1. README's sickle-cell row (86-image test set).** One of the three flips, `data/raw/Positive/Unlabelled/306.jpg`, is in the test split: TensorFlow reference 0.5220 (correct, positive) -> phone (real, measured) 0.4765 (incorrect, negative).

| | Accuracy | Sensitivity | Specificity |
|---|---|---|---|
| README today (computed with TensorFlow preprocessing) | 94.19% | 93.75% | 95.45% |
| **What the app actually produces on this phone (measured)** | **93.02%** | **92.19%** | **95.45% (unchanged)** |

TP 60->59, FN 4->5, TN/FP unchanged. For context (not measured, an estimate using the validated resize emulation on TensorFlow-decoded pixels): the OLD, pre-Phase-14d Android resize would likely have scored this same image at ~0.534 (still correct) -- so this is not "the fix made this image wrong", it is that the image's true model output (0.522) is so close to 0.5 that both the old resize bug and the residual decoder noise can flip it, in either direction, independently of each other.

**2. `phase14b_guardrail_v3_results.md`'s "120 official BBBC test images" row.** Reported 111/120 (92.5%) accepted was computed with TensorFlow preprocessing. Measured on the phone: **110/120 (91.7%) accepted** (1 image, TF 0.5198 -> phone 0.4709, crosses the same 0.5 boundary). The 163-field, 86-sickle, 387-NIH, 945-held-out and 986-fresh guardrail numbers all showed 0 flips on-device and are unaffected.

## Numbers confirmed UNCHANGED
- The malaria (BBBC041) classifier's own reported numbers (`malaria_bbbc041_results.md`, and the Phase 11 threshold-validation figures) are computed by scripts that feed precomputed 224x224 crop arrays directly into the model -- they never call `ImageClassifier`'s resize at all, so they are unaffected by construction, not just by measurement (confirmed 0.000000 diff on 200 such crops, but note that check is an identity resize and doesn't exercise the resize code path).
- Guardrail: 163 BBBC fields, 86 sickle test images, 387 NIH crops, 945 held-out non-smear, 986 fresh non-smear: 0 flips, max diff <=0.057, all consistent with decoder-only noise.
- README's malaria row (96.30%/94.34%/98.26%) is the archived NIH model's numbers and is unrelated to and unaffected by this change (it was already stale/mismatched with the shipped model for a separate, pre-existing reason flagged earlier).

## Still open (not fabricated, flagged as a gap)
No large native-resolution single-cell malaria photo set with ground truth was available to test the resize kernel specifically under heavy downscaling for that model (the only "malaria" test images available are already close to 224px, so shrinking is mild or an upscale -- a different regime from the sickle field photos, where this kernel difference is visible). The malaria classifier was trained on OpenCV-cropped-and-resized boxes (`build_crops.py`), not on `tf.image.resize`'d whole photos, so a heavily-downscaled whole-photo single-cell capture is an untested scenario for the resize-kernel question specifically; all measured evidence so far (NIH crops, BBBC crops) shows zero effect, but those are not stress tests of this particular concern.

## Recommendation
Update `README.md`'s sickle-cell row and `phase14b_guardrail_v3_results.md`'s 120-image row to the phone-measured numbers above (or state both the desktop/reference and on-device numbers side by side) before finalizing anything downstream -- these are small changes, but real ones, and were requested to be flagged rather than silently corrected. I have not edited either document's headline numbers myself.
