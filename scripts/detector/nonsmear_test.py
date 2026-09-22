"""
Phase 13 safety check: what does the DEPLOYED wide-field pipeline do on genuinely non-smear photos?
Runs letterbox 640 -> float16 TFLite detector (score >= 0.3, 3x3 peak pick) -> crops -> bundled malaria classifier -> "any cell >= 0.985",
exactly as the app would, on images from the guardrail's own negative data (faces, everyday objects, scenes, textures, screenshots,
documents, synthetic capture failures), up to 60 per category (fixed seed). Per image the app would show one of:
  Inconclusive (zero cells found) / Negative (cells found, none >= 0.985) / Positive (some cell >= 0.985).
Images stay local (third-party, gitignored). Usage: python scripts/detector/nonsmear_test.py
"""
import glob
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join("scripts", "bbbc041"))
sys.path.insert(0, os.path.dirname(__file__))
import build_crops
from export_tflite import decode_prob, letterbox
from verify_tflite_pipeline import ASSET, Tfl
from common import read_bgr

ROOT = os.path.join("data", "guardrail", "negatives")
T_FIELD, DET_THR, N = 0.985, 0.3, 60

if __name__ == "__main__":
    det = Tfl(path=os.path.join("model_output", "detector", "cell_detector_float16.tflite"))
    cls = Tfl(path=ASSET)
    build_crops.PAD = 1.15
    rng = random.Random(0)
    print(f"{'category':18s} {'n':>3s} | {'Inconclusive':>12s} {'Negative':>9s} {'Positive':>9s} | cells found: median / max | per-cell >=0.5 | example top scores of images with cells")
    all_pos = []
    per_image = []
    for cat in sorted(os.listdir(ROOT)):
        files = sorted(glob.glob(os.path.join(ROOT, cat, "*")))
        files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))]
        files = rng.sample(files, min(N, len(files)))
        res, ncell, tops, cell_scores = [], [], [], []
        for f in files:
            img = read_bgr(f)
            if img is None:
                continue
            x, s = letterbox(img)
            boxes, sc = decode_prob(det(x[None])[0], DET_THR)
            boxes = boxes / s
            ps = []
            for b in boxes:
                c = build_crops.crop(img, {"minimum": {"r": b[1], "c": b[0]}, "maximum": {"r": b[3], "c": b[2]}})
                ps.append(float(cls(((c.astype(np.float32) - 127.5) / 127.5)[None]).reshape(-1)[0]))
            ncell.append(len(ps)); cell_scores += ps
            per_image.append((cat, os.path.basename(f), len(ps), max(ps) if ps else 0.0))
            if not ps:
                res.append("Inconclusive")
            else:
                top = max(ps); tops.append(top)
                res.append("Positive" if top >= T_FIELD else "Negative")
                if top >= T_FIELD:
                    all_pos.append((cat, os.path.basename(f), len(ps), top))
        n = len(res)
        cs = np.array(cell_scores) if cell_scores else np.zeros(0)
        print(f"{cat:18s} {n:3d} | {res.count('Inconclusive'):12d} {res.count('Negative'):9d} {res.count('Positive'):9d} | "
              f"{np.median(ncell):.0f} / {max(ncell)} | {((cs >= .5).sum() if len(cs) else 0)}/{len(cs)} | {np.round(sorted(tops)[::-1][:4], 3).tolist()}")
    print("\nImages that would show POSITIVE:", all_pos if all_pos else "none")
    with open(os.path.join('model_output', 'detector', 'phase13_nonsmear_per_image.csv'), 'w') as fh:
        fh.write("category,file,cells,top_score\n")
        for r in per_image:
            fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.5f}\n")
