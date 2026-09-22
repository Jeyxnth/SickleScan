"""
Train a classifier variant with EXACTLY the baseline recipe (scripts/bbbc041/train.py: same augmentation, model, phases,
seed, early stopping, balanced class weights), only the data differs.
Usage: python scripts/classifier_v2/train_variant.py NAME TRAIN_NPZ[,EXTRA_NPZ...] VAL_NPZ
Saves artifacts/classifier_v2/NAME/model.keras
"""
import importlib.util
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

spec = importlib.util.spec_from_file_location("bbbc_train", os.path.join("scripts", "bbbc041", "train.py"))
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


def load(p):
    z = np.load(p, allow_pickle=True)
    return z["X"], z["y"].astype(np.float32)


if __name__ == "__main__":
    name, train_paths, val_path = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
    out = os.path.join("artifacts", "classifier_v2", name)
    os.makedirs(out, exist_ok=True)
    tf.keras.utils.set_random_seed(42)
    parts = [load(p) for p in train_paths]
    Xtr = np.concatenate([p[0] for p in parts]); ytr = np.concatenate([p[1] for p in parts])
    Xva, yva = load(val_path)
    w = compute_class_weight("balanced", classes=np.array([0.0, 1.0]), y=ytr)
    cw = {0: w[0], 1: w[1]}
    print(name, "train", Xtr.shape, "infected frac", ytr.mean(), "val", Xva.shape, "class weights", cw, flush=True)
    train_ds, val_ds = B.make_train(Xtr, ytr), B.make_eval(Xva, yva)
    model, base = B.build()

    def cbs(f):
        return [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(os.path.join(out, f), monitor="val_loss", save_best_only=True)]

    B.compile_(model, 1e-3)
    model.fit(train_ds, validation_data=val_ds, epochs=20, class_weight=cw, callbacks=cbs("head.keras"), verbose=2)
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    B.compile_(model, 1e-5)
    model.fit(train_ds, validation_data=val_ds, epochs=15, class_weight=cw, callbacks=cbs("ft.keras"), verbose=2)
    model.save(os.path.join(out, "model.keras"))
    print("saved", out, flush=True)
