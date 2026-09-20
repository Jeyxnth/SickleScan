"""
Guardrail training: MobileNetV2 transfer learning, frozen-head then
fine-tune the top 30 base layers at lr 1e-5, early stopping on val_loss
(patience 5) -- same recipe as the disease models. Class weights because the
classes are ~9% imbalanced after adding the synthetic capture failures.

Trains ONE head variant per run so both can be compared:
  --head dense128 : GAP -> Dense(128) -> Dropout(0.3) -> sigmoid  (disease-model head)
  --head simple   : GAP -> Dropout(0.2) -> sigmoid                (lighter head)
Saves artifacts/guardrail/<head>/guardrail_keras.keras + history.json
"""
import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import IMG_SIZE, load_split, make_eval_dataset, make_train_dataset

BATCH_SIZE = 32
HEAD_EPOCHS = 20
FINETUNE_EPOCHS = 15
FINETUNE_LAYERS = 30


def build_model(head):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if head == "dense128":
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
    else:
        x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs), base


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=["dense128", "simple"], required=True)
    args = ap.parse_args()
    out_dir = os.path.join("artifacts", "guardrail", args.head)
    os.makedirs(out_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(42)

    Xtr, ytr, _, _ = load_split("train", "uint8")
    Xva, yva, _, _ = load_split("val", "float32")
    train_ds = make_train_dataset(Xtr, ytr, BATCH_SIZE)
    val_ds = make_eval_dataset(Xva, yva, BATCH_SIZE)
    print(f"Train: {len(Xtr)}  Val: {len(Xva)}  head={args.head}")

    w = compute_class_weight("balanced", classes=np.array([0.0, 1.0]), y=ytr)
    class_weights = {0: w[0], 1: w[1]}
    print("Class weights (0=not_smear, 1=smear):", class_weights)

    model, base = build_model(args.head)

    def cbs(name):
        return [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(os.path.join(out_dir, name), monitor="val_loss", save_best_only=True),
        ]

    print("\n=== Phase 1: head (base frozen) ===")
    compile_model(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS,
                   class_weight=class_weights, callbacks=cbs("best_head.keras"))

    print("\n=== Phase 2: fine-tune top layers ===")
    base.trainable = True
    for layer in base.layers[:-FINETUNE_LAYERS]:
        layer.trainable = False
    compile_model(model, 1e-5)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=FINETUNE_EPOCHS,
                   class_weight=class_weights, callbacks=cbs("best_finetuned.keras"))

    model.save(os.path.join(out_dir, "guardrail_keras.keras"))
    hist = {}
    for h in (h1.history, h2.history):
        for k, v in h.items():
            hist.setdefault(k, []).extend(v)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    print("Saved", out_dir)


if __name__ == "__main__":
    main()
