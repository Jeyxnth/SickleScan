"""
Phase 14: export guardrail v2 to TFLite (float16 like v1) and verify fidelity against the Keras v2 model on every evaluation set
(same images as evaluate_v2.py). Writes model_output/guardrail/guardrail_v2_model.tflite (NOT copied into the app).
Usage: python scripts/guardrail_v2/convert_v2.py
"""
import csv
import glob
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
import evaluate_v2 as ev  # (load() and FRESH only)
import data_pipeline as dp

VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"
ART = os.path.join("artifacts", f"guardrail_{VERSION}", "dense128")
OUT = os.path.join("model_output", "guardrail")


def convert(kind):
    c = tf.lite.TFLiteConverter.from_saved_model(os.path.join(ART, "saved_model"))
    if kind == "float16":
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.target_spec.supported_types = [tf.float16]
    return c.convert()


def run(blob, X):
    it = tf.lite.Interpreter(model_content=blob); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    out = []
    for x in dp.normalize(X):
        it.set_tensor(i["index"], x[None].astype(np.float32)); it.invoke()
        out.append(it.get_tensor(o["index"]).reshape(-1)[0])
    return np.array(out)


if __name__ == "__main__":
    model = tf.keras.models.load_model(os.path.join(ART, "guardrail_keras.keras"))
    model.export(os.path.join(ART, "saved_model"))
    blobs = {k: convert(k) for k in ("float16", "float32")}
    open(os.path.join(OUT, f"guardrail_{VERSION}_model.tflite"), "wb").write(blobs["float16"])
    print({k: f"{len(b) / 1e6:.2f} MB" for k, b in blobs.items()})
    rows = list(csv.DictReader(open(os.path.join("data", "guardrail", "manifest.csv"), newline="", encoding="utf-8")))
    evrows = list(csv.DictReader(open(os.path.join("data", "guardrail", "bbbc_eval.csv"), newline="", encoding="utf-8")))
    paths = ([r["path"] for r in evrows] + [r["path"] for r in rows if r["split"] == "test"]
             + [r["path"] for r in rows if r["label"] == "not_smear" and r["split"] == "val"]
             + sorted(glob.glob(os.path.join(ev.FRESH, "*", "*.jpg"))))
    X = ev.load(paths)
    kp = model.predict(dp.normalize(X).astype(np.float32), batch_size=64, verbose=0).reshape(-1)
    print(f"{len(X)} images (163 BBBC fields, 120 BBBC test, guardrail test split, non-smear val split, 986 fresh non-smear)")
    for k, b in blobs.items():
        p = run(b, X)
        flips = int(((p >= .5) != (kp >= .5)).sum())
        print(f"  {k}: decisions differing from Keras at 0.5: {flips}/{len(X)}; max |P diff| {np.abs(p - kp).max():.4f}, mean {np.abs(p - kp).mean():.6f}")
