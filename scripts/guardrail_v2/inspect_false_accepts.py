"""Phase 14: what does v2 falsely accept, and what would the rest of the wide-field pipeline (detector + classifier + 20-cell gate) do with those images?
Usage: python scripts/guardrail_v2/inspect_false_accepts.py"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join("scripts", "bbbc041"))
sys.path.insert(0, os.path.join("scripts", "detector"))
import build_crops
from export_tflite import decode_prob, letterbox
from verify_tflite_pipeline import ASSET, Tfl
from common import read_bgr

PATHS = [
    ("held-out", "data/guardrail/negatives/coco/0658.jpg", 0.979, 0.977),
    ("held-out", "data/guardrail/negatives/dtd/0133.jpg", 0.972, 0.011),
    ("fresh", "data/guardrail/fresh_negatives/dtd/0047.jpg", 0.644, 0.000),
    ("fresh", "data/guardrail/fresh_negatives/dtd/0054.jpg", 0.866, 0.000),
    ("fresh", "data/guardrail/fresh_negatives/dtd/0139.jpg", 0.575, 0.000),
    ("fresh", "data/guardrail/fresh_negatives/dtd/0168.jpg", 0.913, 0.447),
]

if __name__ == "__main__":
    det = Tfl(path=os.path.join("model_output", "detector", "cell_detector_float16.tflite"))
    cls = Tfl(path=ASSET)
    build_crops.PAD = 1.15
    tiles = []
    for tag, p, p2, p1 in PATHS:
        img = read_bgr(p)
        x, s = letterbox(img)
        boxes, sc = decode_prob(det(x[None])[0], 0.3)
        boxes = boxes / s
        ps = [float(cls(((build_crops.crop(img, {"minimum": {"r": b[1], "c": b[0]}, "maximum": {"r": b[3], "c": b[2]}}).astype(np.float32) - 127.5) / 127.5)[None]).reshape(-1)[0]) for b in boxes]
        cells, top = len(ps), (max(ps) if ps else 0.0)
        ans = "Inconclusive (gate)" if cells < 20 else ("Positive" if top >= 0.985 else "Negative")
        print(f"{tag:8s} {p:52s} v2 P(smear) {p2:.3f} | detector cells {cells:3d}, top cell score {top:.3f} -> pipeline answer: {ans}")
        t = cv2.resize(img, (240, 240)); cv2.putText(t, f"v2 {p2:.2f} cells {cells}", (4, 18), 0, 0.55, (0, 255, 255), 2)
        tiles.append(t)
    cv2.imencode(".png", np.hstack(tiles))[1].tofile(os.path.join("model_output", "guardrail", "phase14_false_accepts.png"))
