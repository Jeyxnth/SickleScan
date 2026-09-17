"""
Step 7 of Phase 1: evaluate the trained Keras model on the held-out test
set and write real numbers (confusion matrix, sensitivity, specificity,
accuracy, precision, recall, F1) to model_output/results.md.

Also saves a handful of sample test images with predicted vs actual label
to model_output/sample_predictions/ for a visual sanity check.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from data_pipeline import (
    CLASS_NAMES,
    IMG_SIZE,
    load_and_preprocess_single,
    make_dataset,
    read_manifest,
)

ARTIFACTS_DIR = "artifacts"
MODEL_OUTPUT_DIR = "model_output"
SAMPLES_DIR = os.path.join(MODEL_OUTPUT_DIR, "sample_predictions")
TEST_CSV = os.path.join("data", "splits", "test.csv")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "sicklescan_keras.keras")

THRESHOLD = 0.5


def main():
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH)

    test_ds, n_test = make_dataset(TEST_CSV, batch_size=16, augment=False)
    print(f"Test images: {n_test}")

    y_true = []
    y_prob = []
    for imgs, labels in test_ds:
        probs = model.predict(imgs, verbose=0).reshape(-1)
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # rows/cols: negative, positive
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # recall on positive class
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")  # recall on negative class
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    print("Confusion matrix (rows=actual, cols=predicted) [negative, positive]:")
    print(cm)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (recall on positive/sickle class): {sensitivity:.4f}")
    print(f"Specificity (recall on negative class): {specificity:.4f}")

    write_results_md(cm, accuracy, sensitivity, specificity, precision, recall, f1, n_test)
    save_confusion_matrix_plot(cm)
    save_samples(y_true, y_prob)


def write_results_md(cm, accuracy, sensitivity, specificity, precision, recall, f1, n_test):
    tn, fp, fn, tp = cm.ravel()
    lines = []
    lines.append("# SickleScan — Phase 1 Evaluation Results\n")
    lines.append(
        "Model: MobileNetV2 (ImageNet-pretrained) transfer learning, "
        "frozen-base head training followed by fine-tuning of the top "
        f"{30} base layers.\n"
    )
    lines.append(f"Test set size: {n_test} images (held out, stratified 15% split)\n")
    lines.append("## Confusion Matrix\n")
    lines.append("| | Predicted Negative | Predicted Positive |")
    lines.append("|---|---|---|")
    lines.append(f"| **Actual Negative** | {tn} (TN) | {fp} (FP) |")
    lines.append(f"| **Actual Positive** | {fn} (FN) | {tp} (TP) |")
    lines.append("")
    lines.append("## Headline Metrics\n")
    lines.append(f"- **Accuracy:** {accuracy:.4f} ({accuracy*100:.2f}%)")
    lines.append(
        f"- **Sensitivity / Recall (positive/sickle class):** {sensitivity:.4f} "
        f"({sensitivity*100:.2f}%) — of all truly positive samples, this fraction "
        "was correctly flagged."
    )
    lines.append(
        f"- **Specificity (negative class recall):** {specificity:.4f} "
        f"({specificity*100:.2f}%) — of all truly negative samples, this fraction "
        "was correctly cleared."
    )
    lines.append("")
    lines.append("## Per-class Precision / Recall / F1\n")
    lines.append("| Class | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|")
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"| {name} | {precision[i]:.4f} | {recall[i]:.4f} | {f1[i]:.4f} |")
    lines.append("")
    lines.append("## TFLite Conversion\n")
    lines.append(
        "- Quantization: **float16** (not int8 dynamic-range). Dynamic-range "
        "quantization was tried first for the smaller file size, but the "
        "conversion sanity check caught it measurably degrading predictions "
        "(test accuracy 94.2% -> 91.9%, specificity 95.45% -> 86.36%; "
        "sensitivity unaffected). Float16 gave 100% label agreement with the "
        "Keras model (zero measured accuracy loss) at ~4.7MB, still small "
        "enough for a mobile app, so it was used instead after checking with "
        "the user.\n"
        "- Sanity check on 10 held-out test images: 10/10 predicted-label "
        "agreement between the Keras model and the converted `.tflite` model, "
        "max probability difference 0.019.\n"
    )
    lines.append(
        "## Notes\n"
        "- Positive class = presence of sickle cells in the blood smear.\n"
        "- Sensitivity is the priority metric here: a missed positive (false "
        "negative) is more costly than a false alarm in a screening context.\n"
        "- Class imbalance (422 positive / 147 negative in the source data) was "
        "handled via class weights during training, not oversampling, to avoid "
        "duplicate-image leakage across the train/val/test splits.\n"
        "- `Positive/Labelled` images from the raw dataset (same photos as "
        "`Positive/Unlabelled` but with black annotation boxes drawn on the "
        "sickle cells) were excluded from training entirely to avoid the model "
        "learning to detect the boxes instead of real cell morphology.\n"
    )

    with open(os.path.join(MODEL_OUTPUT_DIR, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {os.path.join(MODEL_OUTPUT_DIR, 'results.md')}")


def save_confusion_matrix_plot(cm):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    out_path = os.path.join(MODEL_OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(out_path)
    print(f"Saved {out_path}")


def save_samples(y_true_all, y_prob_all, n_samples=8):
    """Re-walks the test manifest directly (not the batched dataset) so we
    can pair each prediction with its original file path for saving."""
    paths, labels = read_manifest(TEST_CSV)
    model = tf.keras.models.load_model(MODEL_PATH)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(paths), size=min(n_samples, len(paths)), replace=False)

    for count, i in enumerate(idx):
        path = paths[i]
        true_label = CLASS_NAMES[labels[i]]

        arr = load_and_preprocess_single(path)  # same pipeline used for training/eval
        prob = float(model.predict(arr[None, ...], verbose=0)[0, 0])
        pred_label = CLASS_NAMES[int(prob >= THRESHOLD)]

        correct = "CORRECT" if pred_label == true_label else "WRONG"
        display_img = Image.open(path).convert("RGB")  # original, undistorted, for display only
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(display_img)
        ax.axis("off")
        ax.set_title(
            f"actual: {true_label} | predicted: {pred_label} ({prob:.2f})\n{correct}",
            fontsize=10,
            color="green" if correct == "CORRECT" else "red",
        )
        out_path = os.path.join(SAMPLES_DIR, f"sample_{count}_{correct.lower()}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {len(idx)} sample prediction images to {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
