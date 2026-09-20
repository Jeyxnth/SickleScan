"""
Evaluate the trained malaria Keras model on the held-out test set --
identical methodology to scripts/evaluate.py (sickle cell): confusion
matrix, accuracy, sensitivity, specificity, per-class precision/recall/F1,
written to model_output/malaria/malaria_results.md, plus sample
predictions saved as images.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import (
    CLASS_NAMES,
    IMG_SIZE,
    load_and_preprocess_single,
    make_dataset,
    read_manifest,
)

ARTIFACTS_DIR = os.path.join("artifacts", "malaria")
MODEL_OUTPUT_DIR = os.path.join("model_output", "malaria")
SAMPLES_DIR = os.path.join(MODEL_OUTPUT_DIR, "sample_predictions")
TEST_CSV = os.path.join("data", "malaria", "splits", "test.csv")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "malaria_keras.keras")

THRESHOLD = 0.5


def main():
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH)

    test_ds, n_test = make_dataset(TEST_CSV, batch_size=32, augment=False)
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

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # rows/cols: uninfected, parasitized
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    print("Confusion matrix (rows=actual, cols=predicted) [uninfected, parasitized]:")
    print(cm)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (recall on parasitized class): {sensitivity:.4f}")
    print(f"Specificity (recall on uninfected class): {specificity:.4f}")

    write_results_md(cm, accuracy, sensitivity, specificity, precision, recall, f1, n_test)
    save_confusion_matrix_plot(cm)
    save_samples(y_true, y_prob)


def write_results_md(cm, accuracy, sensitivity, specificity, precision, recall, f1, n_test, quant_note=None):
    tn, fp, fn, tp = cm.ravel()
    lines = []
    lines.append("# SickleScan — Malaria Model Evaluation Results (Phase 5)\n")
    lines.append(
        "Model: MobileNetV2 (ImageNet-pretrained) transfer learning, "
        "frozen-base head training followed by fine-tuning of the top "
        "30 base layers -- identical architecture/methodology to the sickle "
        "cell model.\n"
    )
    lines.append(f"Test set size: {n_test} images (held out, stratified 15% split)\n")
    lines.append(
        "Dataset: NIH/NLM LHNCBC Malaria Cell Images (Parasitized/Uninfected), "
        "27,558 images, downloaded directly from "
        "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip "
        "(also confirmed live on Kaggle as "
        "iarunava/cell-images-for-detecting-malaria).\n"
    )
    lines.append("## Confusion Matrix\n")
    lines.append("| | Predicted Uninfected | Predicted Parasitized |")
    lines.append("|---|---|---|")
    lines.append(f"| **Actual Uninfected** | {tn} (TN) | {fp} (FP) |")
    lines.append(f"| **Actual Parasitized** | {fn} (FN) | {tp} (TP) |")
    lines.append("")
    lines.append("## Headline Metrics\n")
    lines.append(f"- **Accuracy:** {accuracy:.4f} ({accuracy*100:.2f}%)")
    lines.append(
        f"- **Sensitivity / Recall (parasitized class):** {sensitivity:.4f} "
        f"({sensitivity*100:.2f}%) — of all truly parasitized samples, this "
        "fraction was correctly flagged."
    )
    lines.append(
        f"- **Specificity (uninfected class recall):** {specificity:.4f} "
        f"({specificity*100:.2f}%) — of all truly uninfected samples, this "
        "fraction was correctly cleared."
    )
    lines.append("")
    lines.append("## Per-class Precision / Recall / F1\n")
    lines.append("| Class | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|")
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"| {name} | {precision[i]:.4f} | {recall[i]:.4f} | {f1[i]:.4f} |")
    lines.append("")
    if quant_note:
        lines.append("## TFLite Conversion\n")
        lines.append(quant_note)
    lines.append(
        "## Notes\n"
        "- Positive/parasitized class = malaria parasite visible in the red "
        "blood cell image.\n"
        "- Sensitivity is the priority metric, same as the sickle cell model: "
        "a missed parasitized cell is more costly than a false alarm in a "
        "screening context.\n"
        "- Classes are naturally balanced in the source data (13,779 / "
        "13,779), so no class weighting was actually needed here -- the same "
        "class-weight code path from the sickle cell pipeline was kept for "
        "methodological consistency; the computed weights were ~1.0 each.\n"
        "- Splits are image-level stratified random (matching the sickle "
        "cell methodology and standard practice for this benchmark dataset), "
        "not grouped by source microscope field/patient. Filenames encode a "
        "field/patient ID (e.g. `C100P61ThinF_...`), so cells from the same "
        "field can land in different splits; this is the standard way this "
        "dataset is used in the literature, but is a real (mild) leakage "
        "risk worth naming rather than glossing over.\n"
    )

    with open(os.path.join(MODEL_OUTPUT_DIR, "malaria_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {os.path.join(MODEL_OUTPUT_DIR, 'malaria_results.md')}")


def save_confusion_matrix_plot(cm):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
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
    paths, labels = read_manifest(TEST_CSV)
    model = tf.keras.models.load_model(MODEL_PATH)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(paths), size=min(n_samples, len(paths)), replace=False)

    for count, i in enumerate(idx):
        path = paths[i]
        true_label = CLASS_NAMES[labels[i]]

        arr = load_and_preprocess_single(path)
        prob = float(model.predict(arr[None, ...], verbose=0)[0, 0])
        pred_label = CLASS_NAMES[int(prob >= THRESHOLD)]

        correct = "CORRECT" if pred_label == true_label else "WRONG"
        display_img = Image.open(path).convert("RGB")
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
