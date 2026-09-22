# SickleScan — Phase 1 Evaluation Results

Model: MobileNetV2 (ImageNet-pretrained) transfer learning, frozen-base head training followed by fine-tuning of the top 30 base layers.

Test set size: 86 images (held out, stratified 15% split)

## Confusion Matrix

**On-device (real phone, measured — see footnote¹):**

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 21 (TN) | 1 (FP) |
| **Actual Positive** | 5 (FN) | 59 (TP) |

## Headline Metrics

- **Accuracy:** 0.9302 (93.02%)¹
- **Sensitivity / Recall (positive/sickle class):** 0.9219 (92.19%)¹ — of all truly positive samples, this fraction was correctly flagged.
- **Specificity (negative class recall):** 0.9545 (95.45%) — of all truly negative samples, this fraction was correctly cleared.

¹ Measured on-device (real phone, not desktop) on these same 86 test images. The figure computed with the Python/TensorFlow preprocessing this model was originally validated with is accuracy 0.9419 (94.19%), sensitivity 0.9375 (93.75%), confusion matrix TN 21 / FP 1 / FN 4 / TP 60 — the difference is one boundary-adjacent test image (`306.jpg`) where Android's JPEG decoder produces slightly different pixels than TensorFlow's, not a resize or model issue; see `detector/phase14c_preprocessing_root_cause.md`.

## Per-class Precision / Recall / F1

**On-device (measured):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| negative | 0.8077 | 0.9545 | 0.8750 |
| positive | 0.9833 | 0.9219 | 0.9516 |

Original (Python/TensorFlow preprocessing): negative 0.8400 / 0.9545 / 0.8936; positive 0.9836 / 0.9375 / 0.9600.

## TFLite Conversion

- Quantization: **float16** (not int8 dynamic-range). Dynamic-range quantization was tried first for the smaller file size, but the conversion sanity check caught it measurably degrading predictions (test accuracy 94.2% -> 91.9%, specificity 95.45% -> 86.36%; sensitivity unaffected). Float16 gave 100% label agreement with the Keras model (zero measured accuracy loss) at ~4.7MB, still small enough for a mobile app, so it was used instead after checking with the user.
- Sanity check on 10 held-out test images: 10/10 predicted-label agreement between the Keras model and the converted `.tflite` model, max probability difference 0.019.

## Notes
- Positive class = presence of sickle cells in the blood smear.
- Sensitivity is the priority metric here: a missed positive (false negative) is more costly than a false alarm in a screening context.
- Class imbalance (422 positive / 147 negative in the source data) was handled via class weights during training, not oversampling, to avoid duplicate-image leakage across the train/val/test splits.
- `Positive/Labelled` images from the raw dataset (same photos as `Positive/Unlabelled` but with black annotation boxes drawn on the sickle cells) were excluded from training entirely to avoid the model learning to detect the boxes instead of real cell morphology.
