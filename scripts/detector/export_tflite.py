"""
Phase 13 Task 1: export the CenterNet cell detector to TFLite (float16 weights, like the other models) and verify it.

Deployment form: FIXED 640x640 input (TFLite needs a static shape). The app letterboxes every photo the same way:
longer side -> 640 px (area-average when shrinking, bilinear when enlarging), pasted top-left on a white 640x640 canvas,
scaled to [-1,1]. Output (1,160,160,5) = [sigmoid(heat), w, h, dx, dy] (stride 4); the sigmoid is inside the model so the
app only does the 3x3 peak pick. The Keras reference for every comparison is the SAME wrapped model (fixed 640 input).
Writes model_output/detector/cell_detector_{float16,float32}.tflite and prints fidelity numbers.
Usage: python scripts/detector/export_tflite.py
"""
import os
import random
import sys

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join("scripts", "bbbc041"))
sys.path.insert(0, os.path.dirname(__file__))
import train as T
from common import decode, image_path, iou_matrix, read_bgr, splits, INFECTED

SIZE = 640
OUT = os.path.join("model_output", "detector")
SAVED = os.path.join("artifacts", "detector", "saved_model_640")


def letterbox(img_bgr):
    """-> float32 (640,640,3) in [-1,1] and the scale used (native px -> letterbox px)."""
    s = SIZE / max(img_bgr.shape[:2])
    w, h = int(round(img_bgr.shape[1] * s)), int(round(img_bgr.shape[0] * s))
    r = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.full((SIZE, SIZE, 3), 255, np.uint8)
    canvas[:h, :w] = r
    return (cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5, s


def wrapped_keras():
    m = tf.keras.models.load_model(os.path.join("artifacts", "detector", "detector_best.keras"),
                                   custom_objects={"loss_fn": T.loss_fn}, compile=False)
    inp = tf.keras.Input(shape=(SIZE, SIZE, 3))
    y = m(inp)
    out = tf.keras.layers.Concatenate()([tf.keras.layers.Activation("sigmoid")(y[..., 0:1]), y[..., 1:5]])
    return tf.keras.Model(inp, out)


def decode_prob(o, thr):
    """o: (160,160,5) with heat already a probability -> reuse common.decode by converting back to a logit."""
    o = o.copy()
    p = np.clip(o[..., 0], 1e-7, 1 - 1e-7)
    o[..., 0] = np.log(p / (1 - p))
    return decode(o, score_thr=thr)


def convert(kind):
    conv = tf.lite.TFLiteConverter.from_saved_model(SAVED)
    if kind == "float16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    return conv.convert()


def run_tflite(blob, xs):
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    outs = []
    for x in xs:
        it.set_tensor(i["index"], x[None])
        it.invoke()
        outs.append(it.get_tensor(o["index"])[0])
    return np.stack(outs)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    km = wrapped_keras()
    km.export(SAVED)     # avoids the TF2.16 / Keras 3 from_keras_model MLIR bug, same as the other exports
    blobs = {k: convert(k) for k in ("float16", "float32")}
    for k, b in blobs.items():
        open(os.path.join(OUT, f"cell_detector_{k}.tflite"), "wb").write(b)
        print(f"{k}: {len(b) / 1e6:.2f} MB")

    sp = splits()["val"]
    inf = [r for r in sp if any(o["category"] in INFECTED for o in r["objects"])]
    neg = [r for r in sp if not any(o["category"] in INFECTED or o["category"] == "difficult" for o in r["objects"])]
    rng = random.Random(1)
    recs = rng.sample(inf, 20) + rng.sample(neg, 20)      # validation fields only (never the reserved photos)
    xs = np.stack([letterbox(read_bgr(image_path(r)))[0] for r in recs])
    ref = km.predict(xs, batch_size=4, verbose=0)
    print(f"\nFidelity vs Keras on {len(recs)} validation fields (20 infected + 20 negative), 640x640 letterbox:")
    for k, b in blobs.items():
        t = run_tflite(b, xs)
        d = np.abs(t - ref)
        same_count = matched = tot = 0
        for a_, b_ in zip(ref, t):
            ba, sa = decode_prob(a_, 0.3); bb, sb = decode_prob(b_, 0.3)
            same_count += len(ba) == len(bb)
            iou = iou_matrix(ba, bb)
            matched += int((iou.max(1) >= 0.9).sum()) if iou.size else 0
            tot += len(ba)
        print(f"  {k}: heat prob max|diff| {d[..., 0].max():.4f}, mean|diff| {d[..., 0].mean():.6f}; size/offset max|diff| {d[..., 1:].max():.4f}; "
              f"fields with identical detection count {same_count}/{len(recs)}; Keras detections re-found by TFLite (IoU>=0.9) {matched}/{tot} = {matched / max(tot, 1):.2%}")
