"""
Phase 14: retrain the guardrail with BBBC041 wide-field images added to the smear class. IDENTICAL recipe, seed, augmentation, head
(dense128), early stopping and class weights to scripts/guardrail/train.py; only the manifest differs (manifest_v2.csv) and outputs go
to artifacts/guardrail_v2/ so the Phase 7 model and caches are untouched.
Usage: python scripts/guardrail_v2/train_v2.py [v2|v3]
"""
import importlib.util
import json
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

G = os.path.join("scripts", "guardrail")
sys.path.insert(0, G)
import data_pipeline as dp

VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"   # "v2" = wide-field images added; "v3" = plus BBBC041 single-cell crops
dp.MANIFEST = os.path.join("data", "guardrail", f"manifest_{VERSION}.csv")
dp.CACHE_DIR = os.path.join("artifacts", f"guardrail_{VERSION}")
spec = importlib.util.spec_from_file_location("g_train", os.path.join(G, "train.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

if __name__ == "__main__":
    out_dir = os.path.join("artifacts", f"guardrail_{VERSION}", "dense128")
    os.makedirs(out_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(42)
    Xtr, ytr, src_tr, _ = dp.load_split("train", "uint8")
    Xva, yva, _, _ = dp.load_split("val", "float32")
    train_ds = dp.make_train_dataset(Xtr, ytr, T.BATCH_SIZE)
    val_ds = dp.make_eval_dataset(Xva, yva, T.BATCH_SIZE)
    print(f"Train: {len(Xtr)} (BBBC041 fields: {int((src_tr == 'bbbc041_field').sum())}, BBBC041 single-cell crops: {int((src_tr == 'bbbc041_cell').sum())})  Val: {len(Xva)}", flush=True)
    w = compute_class_weight("balanced", classes=np.array([0.0, 1.0]), y=ytr)
    cw = {0: w[0], 1: w[1]}
    print("Class weights (0=not_smear, 1=smear):", cw, flush=True)
    model, base = T.build_model("dense128")

    def cbs(name):
        return [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(os.path.join(out_dir, name), monitor="val_loss", save_best_only=True)]

    print("\n=== Phase 1: head (base frozen) ===", flush=True)
    T.compile_model(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=T.HEAD_EPOCHS, class_weight=cw, callbacks=cbs("best_head.keras"), verbose=2)
    print("\n=== Phase 2: fine-tune top layers ===", flush=True)
    base.trainable = True
    for layer in base.layers[:-T.FINETUNE_LAYERS]:
        layer.trainable = False
    T.compile_model(model, 1e-5)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=T.FINETUNE_EPOCHS, class_weight=cw, callbacks=cbs("best_finetuned.keras"), verbose=2)
    model.save(os.path.join(out_dir, "guardrail_keras.keras"))
    hist = {}
    for h in (h1.history, h2.history):
        for k, v in h.items():
            hist.setdefault(k, []).extend(float(x) for x in v)
    json.dump(hist, open(os.path.join(out_dir, "history.json"), "w"), indent=2)
    print("Saved", out_dir, flush=True)
