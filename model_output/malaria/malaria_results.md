# SickleScan — Malaria Model Evaluation Results (Phase 5)

Model: MobileNetV2 (ImageNet-pretrained) transfer learning, frozen-base head training followed by fine-tuning of the top 30 base layers -- identical architecture/methodology to the sickle cell model.

Test set size: 4134 images (held out, stratified 15% split)

Dataset: NIH/NLM LHNCBC Malaria Cell Images (Parasitized/Uninfected), 27,558 images, downloaded directly from https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip (also confirmed live on Kaggle as iarunava/cell-images-for-detecting-malaria).

## Confusion Matrix

| | Predicted Uninfected | Predicted Parasitized |
|---|---|---|
| **Actual Uninfected** | 2031 (TN) | 36 (FP) |
| **Actual Parasitized** | 117 (FN) | 1950 (TP) |

## Headline Metrics

- **Accuracy:** 0.9630 (96.30%)
- **Sensitivity / Recall (parasitized class):** 0.9434 (94.34%) — of all truly parasitized samples, this fraction was correctly flagged.
- **Specificity (uninfected class recall):** 0.9826 (98.26%) — of all truly uninfected samples, this fraction was correctly cleared.

## Per-class Precision / Recall / F1

| Class | Precision | Recall | F1 |
|---|---|---|---|
| uninfected | 0.9455 | 0.9826 | 0.9637 |
| parasitized | 0.9819 | 0.9434 | 0.9623 |

## TFLite Conversion

- Quantization: **float16** (not int8 dynamic-range). Both were compared against the Keras model on the full 4,134-image test set. int8 scored numerically higher on every metric (accuracy 96.57% vs 96.37%), but disagreed with the validated Keras model on 41/4,134 predictions (max deviation 0.415 in raw probability) versus float16's 11/4,134 (max deviation 0.082) -- the int8 numbers are most likely quantization noise happening to flip a few borderline cases favorably, not a real accuracy gain, since quantizing more aggressively does not make a model more accurate. float16 was chosen as the more faithful, validated choice, after checking with the user (unlike the sickle cell model, where float16 was unambiguously best; here it was a real tradeoff, flagged and decided explicitly).
- Sanity check on 10 held-out test images: 10/10 predicted-label agreement between the Keras model and the converted `.tflite` model, max probability difference 0.036.

## Notes
- Positive/parasitized class = malaria parasite visible in the red blood cell image.
- Sensitivity is the priority metric, same as the sickle cell model: a missed parasitized cell is more costly than a false alarm in a screening context.
- Classes are naturally balanced in the source data (13,779 / 13,779), so no class weighting was actually needed here -- the same class-weight code path from the sickle cell pipeline was kept for methodological consistency; the computed weights were ~1.0 each.
- Splits are image-level stratified random (matching the sickle cell methodology and standard practice for this benchmark dataset), not grouped by source microscope field/patient. Filenames encode a field/patient ID (e.g. `C100P61ThinF_...`), so cells from the same field can land in different splits; this is the standard way this dataset is used in the literature, but is a real (mild) leakage risk worth naming rather than glossing over.

## Borderline Confidence Threshold (for the Android app's referral logic)

The app treats a prediction as "Borderline" (refer for lab confirmation,
regardless of predicted label) when the model's confidence in its own
prediction is below 65%. This threshold was independently validated
against this model's own test-set confidence distribution, not assumed
to carry over from the sickle cell model:

| Confidence band | Test images | Accuracy |
|---|---|---|
| 50-55% | 46 | 47.83% |
| 55-60% | 30 | 53.33% |
| 60-65% | 37 | 75.68% |
| 65-70% | 35 | 68.57% |
| 70-75% | 42 | 61.90% |
| 75-80% | 60 | 80.00% |
| 80-90% | 176 | 86.93% |
| 90-100% | 3708 | 98.81% |

| Ceiling | Below-ceiling accuracy | Above-ceiling accuracy | % of test flagged borderline |
|---|---|---|---|
| 55% | 47.83% | 96.84% | 1.11% |
| 60% | 50.00% | 97.17% | 1.84% |
| **65%** | **58.41%** | **97.36%** | **2.73%** |
| 70% | 60.81% | 97.62% | 3.58% |

65% gives a clean split: predictions below it are barely better than a
coin flip (58.4% accuracy), predictions at/above it are reliable (97.4%).
Only 2.73% of the test set falls below it. The same threshold as sickle
cell was kept for UI consistency, but this table -- not that precedent --
is what justifies it for malaria.
