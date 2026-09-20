# SickleScan — Guardrail Classifier Results (Phase 7)

A binary "does this look like a blood smear photo?" pre-check that runs before either disease model. Output = P(smear); the app rejects (soft warning) when P(smear) < 0.5.

## What this model is — and is not (read first)

**Known, accepted limitation: no hard negatives.** No other stained microscopy or histology images were included in the not-smear class. The guardrail therefore detects *"does this image match the training distribution of smear-like photos"* (sickle-cell field photos + malaria cell crops) *versus everyday photos, faces, screenshots, documents, textures and blank/blurred/finger-covered captures*. It does **not** detect *"is this specifically a blood smear versus other microscopy images."* A stained tissue slide, another cell type, or any other microscope image may well be accepted, and this report makes no claim about how such images are handled. Do not read the accuracy numbers below as broader robustness than that.

## Dataset (6,598 images: 3,149 smear / 3,449 not-smear)

Split 70/15/15 stratified by source, seed 42 (train 4,618 / val 990 / test 990). Class weights used (classes are 9% imbalanced).

| Class | Source | Images | Notes |
|---|---|---|---|
| smear | Sickle cell (Kaggle) | 569 | all usable images, both disease labels |
| smear | Malaria (NIH/NLM) | 2,580 | stratified subsample of 27,558 (1,290 + 1,290) so sickle-cell fields aren't 2% of the class |
| not_smear | COCO val2017 | 1,000 | everyday objects, people, scenes |
| not_smear | Places365 val | 700 | scenes / landscapes |
| not_smear | DTD | 450 | textures |
| not_smear | LFW | 400 | faces |
| not_smear | ScreenSpot | 400 | screenshots |
| not_smear | FUNSD | 199 | scanned forms / documents |
| not_smear | **synthetic_black** | 60 | **SYNTHETIC** black frames |
| not_smear | **synthetic_blank** | 60 | **SYNTHETIC** blank / overexposed / flat-colour frames |
| not_smear | **synthetic_blur** | 100 | **SYNTHETIC** heavily blurred real photos |
| not_smear | **synthetic_finger** | 80 | **SYNTHETIC** finger-over-lens look (procedural) |

The 300 synthetic images are procedurally generated (not real captures), kept as separate sources so they are tracked separately, and are a rough stand-in for real capture failures. CIFAR-100 was deliberately not used (32×32 upscaled images would let the model learn "blurry = not smear").

## Model

MobileNetV2 (ImageNet) transfer learning, 224×224, same preprocessing as the disease models, frozen-head training then fine-tune top 30 layers at lr 1e-5, early stopping on val_loss.

Training augmentation targets shortcuts (malaria crops are small, blurry, on black; negatives are sharper/larger): random downscale→upscale blur on all images, black-canvas paste (more often on negatives), black-background recolouring, and — added after the shortcut check below — **round microscope-style field masking and stain-colour tint on negatives.**

**Head comparison (first training round, before the round-field fix; test set):** `dense128` (GAP→Dense128→Dropout→sigmoid, same as the disease models) and `simple` (GAP→Dropout→sigmoid) scored identically — 99.80% accuracy, 1/473 false negatives, 1/517 false positives. No measurable gain from the lighter head, so `dense128` was kept for consistency and is the only head retrained/shipped.

## Test-set results (final model, 990 images, threshold 0.5)

| | Result |
|---|---|
| Accuracy | 99.90% |
| **False-negative rate** (real smear wrongly rejected — "annoying") | **0 / 473 = 0.00%** (sickle 0/86, malaria 0/387) |
| **False-positive rate** (non-smear wrongly accepted — "dangerous") | **1 / 517 = 0.19%** |

These are in-distribution numbers. Real phone photos through a microscope attachment will differ and real rates will be worse; they are unknown. One real sickle-cell image (`pos_155.jpg`, a *training* image) is still rejected at P(smear)=0.23, so the false-reject rate on real smears is not zero (0.3% of the training sickle images).

## Check 2 — false-positive rate BY negative source (test set, 95% Wilson interval)

| Source | Accepted (FP) | Rate | 95% CI |
|---|---|---|---|
| COCO | 1 / 150 | 0.67% | 0.1–3.7% |
| Places365 | 0 / 105 | 0.00% | 0–3.5% |
| DTD | 0 / 67 | 0.00% | 0–5.4% |
| LFW faces | 0 / 60 | 0.00% | 0–6.0% |
| ScreenSpot screenshots | 0 / 60 | 0.00% | 0–6.0% |
| FUNSD documents | 0 / 30 | 0.00% | 0–11.4% |
| synthetic_black | 0 / 9 | 0.00% | 0–29.9% |
| synthetic_blank | 0 / 9 | 0.00% | 0–29.9% |
| synthetic_blur | 0 / 15 | 0.00% | 0–20.4% |
| synthetic_finger | 0 / 12 | 0.00% | 0–24.3% |

No source stands out, but per-source test counts are small: "0 errors" is compatible with true rates of ~5–30% for the smaller sources. The one COCO false positive is `coco/0658.jpg` (P=0.98).

## Check 1 — shortcut learning

Method: hold content fixed, flip the suspected cue, see whether P(smear) follows the cue. Real malaria crops (black background, small, upscaled) score median 1.000 (p10 1.000); real sickle fields median 1.000; real non-smear images median 0.000.

| Probe (non-smear content unless noted) | Accepted |
|---|---|
| Shrunk onto black background | 0.0% |
| Degraded to 64 px (malaria-like blur) | 0.2% |
| Small + black bg + blur together (all malaria cues) | 0.0% |
| Malaria crops with black background **recoloured** (cell untouched) | 100% accepted (stays smear) |
| Malaria crops shrunk onto black canvas | 100% accepted |

These probes reuse transforms the model was trained against, so they show the augmentation worked, not independent proof. **The independent probes found a real shortcut in the first model:**

| Probe (transform NOT in first training run) | 1st model: accepted | Final model: accepted |
|---|---|---|
| Round microscope-style field (black outside a circle) | **14.3%** (blank 100%, finger 100%, black 78%, blur 67%, DTD 45%, screenshots 8%) | 1.4% (DTD 9%, screenshots 2%, synthetic 0%) |
| Round field + pink/purple stain tint | 15.7% | 1.5% |

The first model was partly keying on "round black-cornered field ⇒ smear", which matters because a blank/blurred/finger-covered shot taken *through the microscope attachment* has exactly that round field. I retrained with round-field + stain-tint augmentation on negatives. Because the round field is now in the training distribution, I re-probed with geometry it has never seen:

| Unseen geometry (final model) | Accepted overall | Worst sources |
|---|---|---|
| Off-centre ellipse | 3.1% | DTD 14.9%, synthetic_blank 22% (2/9), synthetic_black 11% (1/9) |
| Inset square (black corners) | 0.2% | none |
| Off-centre ellipse + stain tint | 2.9% | DTD 14.9% |

**Conclusion:** the black-background / small-size / low-resolution shortcut is not what the model relies on. A field-geometry shortcut existed and is much reduced but **not eliminated**: textures (DTD) and flat/blank frames inside an unfamiliar-shaped microscope-style field are still accepted at a few-to-15% rate. The soft-warning UX (below) exists partly for this reason.

## Threshold validation

The default 0.5 is **not a tuned value**. On the validation set every threshold from 0.02 to 0.98 made identical errors (FN ≤1% budget, FP minimized → tied range 0.08–0.98), and only 6 of 990 test predictions fall between 0.02 and 0.98 — the model's outputs are almost entirely 0 or 1. There is nothing on this data to validate a threshold against, so 0.5 (the interpreter default) is reported as un-validated rather than presented as validated.

## TFLite conversion (verified on the full 990-image test set)

| Variant | Size | Accuracy | FNR | FPR | Agreement with Keras | Max abs diff |
|---|---|---|---|---|---|---|
| int8 dynamic | 2.6 MB | 99.90% | 0% | 0.19% | 100% | 0.145 |
| **float16 (shipped)** | **4.7 MB** | 99.90% | 0% | 0.19% | 100% | 0.018 |
| float32 | 9.3 MB | 99.90% | 0% | 0.19% | 100% | ~1e-6 |

int8 matched float16 on this test set (unlike the sickle-cell model, where it measurably hurt), but its output drifts up to 0.145 from Keras; float16 was kept for consistency with the other two models and near-lossless drift. Files: `guardrail_model.tflite`, `labels.txt` (`not_smear`, `smear`).

## Android verification (what was and wasn't run)

- Build verified: `assembleDebug`, `assembleDebugAndroidTest`, and 21 local unit tests pass (incl. the new `GuardrailInterpreterTest` and `GuardrailAggregatorTest`).
- The bundled `guardrail_model.tflite` was run in Python with the app's exact preprocessing on the app's test images: 6/6 non-smear images rejected (photo, scene, face, screenshot, document, finger), and 9 of 10 real smear images (both diseases) accepted; the 10th, `pos_155.jpg`, is the false reject described above.
- `GuardrailClassifierInstrumentedTest` encodes those expectations for a device, but **no emulator/device was available, so it was not executed**; the real on-device flow (camera → guardrail → warning/disease result, Room rejection logging, dashboard) is build-verified only.
