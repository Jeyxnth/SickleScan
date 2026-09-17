"""
Step 5-6 of Phase 1: build a MobileNetV2 transfer-learning classifier and
train it in two phases (frozen head, then fine-tune the top of the base).

Tracks accuracy/precision/recall/F1 on validation each epoch, uses
class weights to handle the imbalance, and early-stops on val_loss.

Saves the best model to artifacts/sicklescan_keras.keras
and the training curves to artifacts/training_history.png
"""
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from data_pipeline import IMG_SIZE, make_dataset, read_manifest

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

TRAIN_CSV = os.path.join("data", "splits", "train.csv")
VAL_CSV = os.path.join("data", "splits", "val.csv")

BATCH_SIZE = 16
HEAD_EPOCHS = 20
FINETUNE_EPOCHS = 15
FINETUNE_LAYERS = 30  # number of trailing base layers to unfreeze


class F1Callback(tf.keras.callbacks.Callback):
    """Adds val_f1 (and train f1) to the logs each epoch, computed from the
    precision/recall metrics Keras already tracks."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for prefix in ("", "val_"):
            p = logs.get(f"{prefix}precision")
            r = logs.get(f"{prefix}recall")
            if p is not None and r is not None:
                f1 = 2 * p * r / (p + r + 1e-7)
                logs[f"{prefix}f1"] = f1
        print(
            f"  -> val_precision={logs.get('val_precision', 0):.4f} "
            f"val_recall={logs.get('val_recall', 0):.4f} "
            f"val_f1={logs.get('val_f1', 0):.4f}"
        )


def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def main():
    train_ds, n_train = make_dataset(
        TRAIN_CSV, batch_size=BATCH_SIZE, augment=True, shuffle=True
    )
    val_ds, n_val = make_dataset(VAL_CSV, batch_size=BATCH_SIZE, augment=False)
    print(f"Train images: {n_train}  Val images: {n_val}")

    _, train_labels = read_manifest(TRAIN_CSV)
    class_weight_values = compute_class_weight(
        class_weight="balanced", classes=np.unique(train_labels), y=train_labels
    )
    class_weights = {i: w for i, w in enumerate(class_weight_values)}
    print("Class weights (0=negative, 1=positive):", class_weights)

    model, base = build_model()

    ckpt_path = os.path.join(ARTIFACTS_DIR, "best_head.keras")
    callbacks_head = [
        F1Callback(),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_loss", save_best_only=True
        ),
    ]

    print("\n=== Phase 1: training classification head (base frozen) ===")
    compile_model(model, lr=1e-3)
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks_head,
    )

    print("\n=== Phase 2: fine-tuning top layers of MobileNetV2 ===")
    base.trainable = True
    # Freeze all but the last FINETUNE_LAYERS layers of the base model
    for layer in base.layers[:-FINETUNE_LAYERS]:
        layer.trainable = False

    ckpt_path_ft = os.path.join(ARTIFACTS_DIR, "best_finetuned.keras")
    callbacks_ft = [
        F1Callback(),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path_ft, monitor="val_loss", save_best_only=True
        ),
    ]

    compile_model(model, lr=1e-5)
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks_ft,
    )

    final_path = os.path.join(ARTIFACTS_DIR, "sicklescan_keras.keras")
    model.save(final_path)
    print(f"\nSaved final model to {final_path}")

    # Persist combined history for plotting / inspection later
    combined_history = {}
    for h in (history1.history, history2.history):
        for k, v in h.items():
            combined_history.setdefault(k, []).extend(v)
    with open(os.path.join(ARTIFACTS_DIR, "history.json"), "w") as f:
        json.dump(combined_history, f, indent=2)

    plot_history(combined_history)


def plot_history(history):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.get("loss", []), label="train_loss")
    axes[0].plot(history.get("val_loss", []), label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    for key in ("accuracy", "val_accuracy", "recall", "val_recall", "f1", "val_f1"):
        if key in history:
            axes[1].plot(history[key], label=key)
    axes[1].set_title("Metrics")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    out_path = os.path.join(ARTIFACTS_DIR, "training_history.png")
    fig.savefig(out_path)
    print(f"Saved training curves to {out_path}")


if __name__ == "__main__":
    main()
