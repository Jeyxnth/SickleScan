"""
End-to-end on annotated BBBC041 TEST images (native scale): heuristic detector (Phase 9a) -> classifier.
Reports (1) detector recall against ground-truth boxes, split by infected vs uninfected cells, and
(2) image-level sensitivity of the full pipeline. Uses the first N test images that contain infected cells.
"""
import json
import os
import random
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join("scripts", "tiling"))
sys.path.insert(0, os.path.dirname(__file__))
from classify_tiles import make_crops
from detect_cells import detect_cells, load_bgr

ROOT = os.path.join("data", "bbbc041", "raw", "malaria")
INF = {"trophozoite", "schizont", "gametocyte", "ring"}
N = 30

if __name__ == "__main__":
    model = tf.keras.models.load_model(os.path.join("artifacts", "bbbc041", "model", "bbbc_keras.keras"))
    test = json.load(open(os.path.join(ROOT, "test.json")))
    rng = random.Random(3)
    recs = [r for r in test if any(o["category"] in INF for o in r["objects"])]
    recs = rng.sample(recs, N)
    hit = {"inf": [0, 0], "rbc": [0, 0]}
    image_p = []
    for rec in recs:
        img = load_bgr(os.path.join(ROOT, rec["image"]["pathname"].lstrip("/")))
        det = detect_cells(img)
        inv = 1 / det["scale"]
        cells, raw, _ = make_crops(det, img)
        cent = [((c["bbox"][0] + c["bbox"][2] / 2) * inv, (c["bbox"][1] + c["bbox"][3] / 2) * inv) for c in cells]
        p = model.predict((np.stack(raw).astype(np.float32) - 127.5) / 127.5, batch_size=64, verbose=0).reshape(-1) if raw else np.zeros(0)
        image_p.append(p)
        for o in rec["objects"]:
            if o["category"] == "difficult" or o["category"] == "leukocyte":
                continue
            b = o["bounding_box"]
            found = any(b["minimum"]["c"] <= x <= b["maximum"]["c"] and b["minimum"]["r"] <= y <= b["maximum"]["r"] for x, y in cent)
            k = "inf" if o["category"] in INF else "rbc"
            hit[k][0] += found
            hit[k][1] += 1
    print(f"Detector recall vs ground truth on {N} test images: infected cells {hit['inf'][0]}/{hit['inf'][1]} = {hit['inf'][0] / hit['inf'][1]:.1%} | "
          f"uninfected RBCs {hit['rbc'][0]}/{hit['rbc'][1]} = {hit['rbc'][0] / hit['rbc'][1]:.1%}")
    for name, fn in (("any cell >= 0.5", lambda q: (q >= .5).any()), ("any cell >= 0.1", lambda q: (q >= .1).any()),
                     (">=3 cells >= 0.5", lambda q: (q >= .5).sum() >= 3)):
        print(f"FULL PIPELINE image-level sensitivity (all {N} images contain infected cells), rule '{name}': {sum(fn(q) for q in image_p)}/{N}")
