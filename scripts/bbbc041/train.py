"""
Binary infected / not-infected classifier on native-scale BBBC041 cell crops.
Same recipe as the earlier disease models: MobileNetV2 (ImageNet) -> GAP -> Dense(128) -> Dropout(0.3) -> sigmoid,
frozen-head training then fine-tune the top 30 base layers at lr 1e-5, early stopping on val_loss (patience 5),
class weights. Preprocessing (x-127.5)/127.5, 224x224.

Augmentation (training only), chosen for the known deployment gap (small, blurry, differently stained field photos):
  flips + 90deg rotations, brightness/contrast, hue + saturation jitter (stain variation across BBBC041 is large,
  some slides are almost greyscale), random grayscale (p=0.1), random downscale->upscale blur (p=0.5).
Saves artifacts/bbbc041/model/bbbc_keras.keras
"""
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

OUT = os.path.join("artifacts", "bbbc041")
MODEL_DIR = os.path.join(OUT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)
BATCH = 32


def load(split):
    z = np.load(os.path.join(OUT, f"crops_{split}.npz"), allow_pickle=True)
    return z["X"], z["y"]


@tf.function
def augment(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    img = tf.image.random_brightness(img, 0.12)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.6, 1.4)
    img = tf.image.random_hue(img, 0.05)
    if tf.random.uniform([]) < 0.1:
        img = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(img))
    if tf.random.uniform([]) < 0.5:
        s = tf.cast(tf.random.uniform([], 32.0, 224.0), tf.int32)
        img = tf.image.resize(tf.image.resize(img, [s, s]), [224, 224])
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img * 2.0 - 1.0, label   # == (x*255-127.5)/127.5


def make_train(X, y):
    ds = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(len(X), seed=42, reshuffle_each_iteration=True)
    return ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)


def make_eval(X, y):
    Xn = (X.astype(np.float32) - 127.5) / 127.5
    return tf.data.Dataset.from_tensor_slices((Xn, y)).batch(BATCH).prefetch(tf.data.AUTOTUNE)


def build():
    base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inp = tf.keras.Input(shape=(224, 224, 3))
    x = base(inp, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out), base


def compile_(m, lr):
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy",
              metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.Precision(name="precision"),
                       tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")])


def main():
    tf.keras.utils.set_random_seed(42)
    Xtr, ytr = load("train")
    Xva, yva = load("val")
    train_ds, val_ds = make_train(Xtr, ytr), make_eval(Xva, yva)
    w = compute_class_weight("balanced", classes=np.array([0.0, 1.0]), y=ytr)
    cw = {0: w[0], 1: w[1]}
    print("train", Xtr.shape, "val", Xva.shape, "class weights", cw, flush=True)
    model, base = build()

    def cbs(name):
        return [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, name), monitor="val_loss", save_best_only=True)]

    print("=== Phase 1: head ===", flush=True)
    compile_(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=20, class_weight=cw, callbacks=cbs("best_head.keras"))
    print("=== Phase 2: fine-tune top 30 layers ===", flush=True)
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    compile_(model, 1e-5)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=15, class_weight=cw, callbacks=cbs("best_finetuned.keras"))
    model.save(os.path.join(MODEL_DIR, "bbbc_keras.keras"))
    hist = {}
    for h in (h1.history, h2.history):
        for k, v in h.items():
            hist.setdefault(k, []).extend(float(x) for x in v)
    json.dump(hist, open(os.path.join(MODEL_DIR, "history.json"), "w"))
    print("saved", MODEL_DIR)


if __name__ == "__main__":
    main()
