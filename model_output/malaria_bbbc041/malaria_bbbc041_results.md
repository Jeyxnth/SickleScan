# SickleScan — Malaria classifier v2 (BBBC041-trained), shipped in Phase 10

This replaces the NIH-trained malaria model **as the bundled app model**. The NIH model and all its results remain archived, unchanged, in `model_output/malaria/` (and in git history).

## What it is
MobileNetV2 (ImageNet) → GAP → Dense(128) → Dropout(0.3) → sigmoid; trained on **native-scale single-cell crops cut from BBBC041 ground-truth boxes** (P. vivax Giemsa thin smears): infected = ring / trophozoite / schizont / gametocyte, not infected = red blood cell + leukocyte. Splits are by image. Full training and validation details: `model_output/bbbc041/phase9b_report.md`. Output convention is unchanged: sigmoid = P(infected), labels `uninfected`, `parasitized` (same ordering, so `labels.txt` did not change).

**Input the app supports: a close-up image of a single blood cell.** There is no cell-detection step (the tiling/detection pipeline from Phase 9 was never integrated), so wide-field photos are classified as a single image and are out of scope.

## Held-out results
| Set | Crops | Accuracy | Infected recall @0.5 | Uninfected specificity |
|---|---|---|---|---|
| Validation (held-out images, same imaging setup as training) | 1,823 (307 infected) | 98.6% | 97.1% | 98.9% |
| **Official test set** (different microscope/resolution, never used for training or selection) | 5,917 (303 infected) | 96.9% | **40.9%** | 99.9% |

**Read this before demoing.** On the shifted test set AUC is still 0.985, but the scores are compressed: infected cells are called uninfected 51% of the time *with ≥65% confidence*. Overall accuracy looks fine only because 95% of crops are uninfected. A fixed 0.5 boundary is therefore not portable across imaging setups: retuned on that set, the same model would reach ~85% sensitivity at ~1% false positives, but only at a threshold of 0.013, and choosing that on the test set would be tuning on the test set. This is unresolved and is a decision for the project owner (see Phase 10 notes).

## Confidence threshold (borderline ceiling), validated on this model's own held-out data
Method as for the earlier models: accuracy of predictions below vs at/above a confidence cutoff (confidence = probability of the predicted class).

| Ceiling | Validation: below / at-above | Test: below / at-above |
|---|---|---|
| 60% | 63.6% (n=11) / 98.8% | 51.6% (n=31) / 97.1% |
| **65% (chosen)** | **65.0% (n=20, 1.1% of crops) / 98.9%** | **50.0% (n=52, 0.9% of crops) / 97.3%** |
| 70% | 55.6% (n=27) / 99.2% | 49.2% (n=65) / 97.4% |
| 90% | 71.2% (n=66, 3.6%) / 99.6% | 47.4% (n=156, 2.6%) / 98.2% |

Below 65% the model is roughly a coin flip (65% / 50%); above it, 97–99% accurate. Higher ceilings flag 2–4× more crops without making the flagged group meaningfully less accurate, so **65% is kept** — the same number as the sickle-cell and NIH-malaria models, validated independently here rather than assumed. The ceiling does not address the sensitivity problem above (most missed infected cells are confidently wrong, not borderline: at 65%, true infected cells on the test set are 33% confident-correct / 16% borderline / **51% confident-wrong**).

## TFLite conversion (verified against Keras on 5,917 test crops + 1,823 val crops + the four real photos)
| Variant | Size | Label agreement with Keras (test / val / photos) | Max abs prob diff (test) |
|---|---|---|---|
| int8 dynamic | 2.6 MB | 99.29% / 99.73% / 100% | 0.63 |
| float16 | 4.7 MB | **99.90%** / 100% / 100% | 0.10 |
| **float32 (shipped)** | 9.3 MB | 100% / 100% / 100% | 0 |

Unlike the earlier models, float16 was *not* lossless here: 6 of 5,917 test predictions flip across 0.5 (all near the boundary; test sensitivity 39.9% vs 40.9%). My acceptance rule was ≥99.9% agreement on the test crops and float16 landed just under it, so **float32 is shipped** (+4.6 MB) so on-device outputs match the validated Python numbers exactly. Switching to float16 is a one-line change if size matters more (see `artifacts`-free reproduction: `scripts/bbbc041/convert_tflite.py`). On the four photos float16 and float32 agree to within 0.004.

## Demo crops (single-cell images for the gallery picker)
`demo_images/malaria/` (local only, gitignored: BBBC041's redistribution licence was not verified), 12 random crops from the **validation** images (held out from training; seeded): 6 infected + 6 uninfected. Shipped model: **12/12 correct** (infected P = 0.99, 1.00, 1.00, 1.00, 0.53, 0.99; uninfected P ≤ 0.009). One infected crop (`infected_5_trophozoite`, P=0.53) sits at 53% confidence, so the app shows it as **Borderline**. Expected values: `demo_images/malaria/expected.csv`.

**Regression to know about:** the new model does not read NIH-style crops (segmented cells on black). The 8 NIH test images bundled for earlier phases all score "uninfected" now — including the 4 truly parasitized ones (P(infected) 0.0001–0.03). The demo must use BBBC041-style crops (or real crops from thin-smear photos), not the old NIH images.

## Other verified behaviour
- Four user photos (whole image, as the app would classify them): 0.335, 0.043, 0.020, 0.078 (float32 Keras vs TFLite identical). These are wide-field / thick-smear-style images and are out of scope (see Phase 9 reports).
- Not verified on a device: no emulator/phone was available in this environment.
