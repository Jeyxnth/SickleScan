"""
Convert the trained guardrail Keras model to TFLite. Same approach as the
disease models: build int8-dynamic-range, float16 and float32 variants,
compare all three to the Keras model on the FULL test set, then pick --
verified, not assumed. Writes model_output/guardrail/{guardrail_model.tflite,
labels.txt} and artifacts/guardrail/quant_comparison.json.

Usage: python scripts/guardrail/convert_tflite.py --head dense128
"""
import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import CLASS_NAMES, load_split, normalize

OUT = os.path.join("model_output", "guardrail")


def build(saved_dir, kind):
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
    if kind == "int8_dynamic":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    elif kind == "float16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    return conv.convert()


def run(tfl_bytes, X):
    interp = tf.lite.Interpreter(model_content=tfl_bytes)
    interp.allocate_tensors()
    i, o = interp.get_input_details()[0], interp.get_output_details()[0]
    out = []
    for x in X:
        interp.set_tensor(i["index"], x[None].astype(np.float32))
        interp.invoke()
        out.append(interp.get_tensor(o["index"]).reshape(-1)[0])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    args = ap.parse_args()
    art = os.path.join("artifacts", "guardrail", args.head)
    model = tf.keras.models.load_model(os.path.join(art, "guardrail_keras.keras"))
    saved = os.path.join(art, "saved_model")
    model.export(saved)  # avoids the TF2.16/Keras3 from_keras_model MLIR bug

    X, y, _, _ = load_split("test", "float32")
    Xn = normalize(X).astype(np.float32)
    kp = model.predict(Xn, batch_size=64, verbose=0).reshape(-1)
    kpred = kp >= 0.5
    results, blobs = {}, {}
    for kind in ("int8_dynamic", "float16", "float32"):
        b = build(saved, kind)
        blobs[kind] = b
        p = run(b, Xn)
        pred = p >= 0.5
        fnr = float(np.mean(~pred[y == 1]))
        fpr = float(np.mean(pred[y == 0]))
        results[kind] = dict(size_kb=len(b) / 1024, accuracy=float(np.mean(pred == (y == 1))), fnr=fnr, fpr=fpr,
                             agreement_with_keras=float(np.mean(pred == kpred)),
                             max_abs_diff=float(np.max(np.abs(p - kp))))
        print(kind, json.dumps(results[kind]))
    print("keras baseline acc", float(np.mean(kpred == (y == 1))))

    # Same rule as the disease models: float16 unless it demonstrably loses to float32 baseline.
    f16 = results["float16"]
    chosen = "float16" if f16["agreement_with_keras"] >= 0.998 else "float32"
    print("chosen:", chosen)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "guardrail_model.tflite"), "wb") as f:
        f.write(blobs[chosen])
    with open(os.path.join(OUT, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(CLASS_NAMES) + "\n")
    with open(os.path.join("artifacts", "guardrail", "quant_comparison.json"), "w") as f:
        json.dump(dict(chosen=chosen, results=results), f, indent=2)
    print("saved", os.path.join(OUT, "guardrail_model.tflite"), f"({len(blobs[chosen]) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
