"""
Full pipeline (Phase 11a Task 4): learned cell detector -> crop each cell (box x 1.15 square, same as the classifier's
training crops) -> EXISTING BBBC041 classifier (unchanged) -> per-image summary.

Usage:
  python scripts/detector/pipeline.py photos     # the four user photos + annotated overlays (kept local)
  python scripts/detector/pipeline.py controls   # 30 malaria-free sickle-dataset fields (same set as Phase 9b)
  python scripts/detector/pipeline.py test       # 30 test images with infected cells (same as Phase 9b) image-level catch
"""
import glob
import os
import random
import sys

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join("scripts", "bbbc041"))
sys.path.insert(0, os.path.dirname(__file__))  # detector dir first: its train.py must win over bbbc041/train.py
import train as T
from common import OUT, decode, image_path, prep_eval, read_bgr, splits
from build_crops import crop

CLS = os.path.join("artifacts", "bbbc041", "model", "bbbc_keras.keras")


def load_models():
    det = tf.keras.models.load_model(os.path.join(OUT, "detector_best.keras"), custom_objects={"loss_fn": T.loss_fn}, compile=False)
    return det, tf.keras.models.load_model(CLS)


def run(det, cls, img, thr):
    x, s = prep_eval(img)
    boxes, sc = decode(det.predict(x, verbose=0)[0], score_thr=thr)
    boxes = boxes / s
    if len(boxes) == 0:
        return boxes, sc, np.zeros(0, np.float32)
    crops = np.stack([crop(img, {"minimum": {"r": b[1], "c": b[0]}, "maximum": {"r": b[3], "c": b[2]}}) for b in boxes])
    p = cls.predict((crops.astype(np.float32) - 127.5) / 127.5, batch_size=64, verbose=0).reshape(-1)
    return boxes, sc, p


def overlay(img, boxes, p, path):
    o = img.copy()
    for b, q in zip(boxes, p):
        col = (0, 0, 255) if q >= 0.5 else ((0, 165, 255) if q >= 0.1 else (0, 200, 0))
        cv2.rectangle(o, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2 if q >= 0.1 else 1)
    cv2.imencode(".png", o)[1].tofile(path)


if __name__ == "__main__":
    mode = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    det, cls = load_models()
    if mode == "photos":
        for f in sorted(glob.glob(os.path.join("data", "tiling_validation", "malaria_positive_fields", "*.*"))):
            if not f.lower().endswith((".jpg", ".webp", ".png", ".jpeg")):
                continue
            img = read_bgr(f)
            b, sc, p = run(det, cls, img, thr)
            print(f"{os.path.basename(f)} {img.shape[1]}x{img.shape[0]}: {len(b)} cells detected | classifier >=0.5: {(p >= .5).sum()}, >=0.1: {(p >= .1).sum()}, "
                  f"max {p.max() if len(p) else 0:.3f} | top scores {np.round(np.sort(p)[::-1][:6], 3).tolist()}")
            overlay(img, b, p, os.path.join("model_output", "detector", f"user_photo_{os.path.splitext(os.path.basename(f))[0]}_pipeline.png"))
    elif mode == "controls":
        fs = sorted(glob.glob("data/raw/Negative/Clear/*.jpg"))[:30]  # same 30 fields as Phase 9b controls
        print(len(fs), "control files")
        n_cells = n05 = 0
        rules = {"any >= 0.5": 0, ">=3 cells >= 0.5": 0, ">=5 cells >= 0.9": 0}
        for f in fs:
            b, sc, p = run(det, cls, read_bgr(f), thr)
            n_cells += len(p); n05 += int((p >= .5).sum())
            rules["any >= 0.5"] += bool((p >= .5).any()); rules[">=3 cells >= 0.5"] += bool((p >= .5).sum() >= 3)
            rules[">=5 cells >= 0.9"] += bool((p >= .9).sum() >= 5)
        print(f"controls: {len(fs)} images, {n_cells} cells, per-cell >=0.5: {n05} = {n05 / max(n_cells, 1):.2%}")
        for k, v in rules.items():
            print(f"  rule '{k}': {v}/{len(fs)} images flagged")
    elif mode == "test":
        inf = {"trophozoite", "schizont", "gametocyte", "ring"}
        recs = [r for r in splits()["test"] if any(o["category"] in inf for o in r["objects"])]
        recs = random.Random(3).sample(recs, 30)
        ps = [run(det, cls, read_bgr(image_path(r)), thr)[2] for r in recs]
        for name, fn in (("any cell >= 0.5", lambda q: (q >= .5).any()), ("any cell >= 0.1", lambda q: (q >= .1).any()),
                         (">=3 cells >= 0.5", lambda q: (q >= .5).sum() >= 3)):
            print(f"FULL PIPELINE image-level sensitivity, rule '{name}': {sum(bool(fn(q)) for q in ps)}/30 "
                  "(heuristic detector: 5/30 at >=0.5, 10/30 at >=0.1; oracle boxes: 74/115 over all infected test images)")
