"""
Annotation-completeness check on a sample of ~20 BBBC041 images:
 (1) overlay GT boxes for visual inspection,
 (2) automated proxy: run the Phase 9a heuristic cell detector, count confidently-detected cells
     whose centre lies inside NO ground-truth box (candidate unannotated cells), and GT boxes with
     no detected cell (detector misses) -- both are noisy, so this is a screening aid, not a verdict.
"""
import json
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join("scripts", "tiling"))
from detect_cells import detect_cells, load_bgr

ROOT = os.path.join("data", "bbbc041", "raw", "malaria")
OUT = os.path.join("artifacts", "bbbc041")
os.makedirs(OUT, exist_ok=True)
COL = {"red blood cell": (0, 200, 0), "leukocyte": (255, 120, 0), "difficult": (0, 255, 255)}
INF = (0, 0, 255)


def draw(img, objs, scale):
    v = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for o in objs:
        b = o["bounding_box"]
        p0 = (int(b["minimum"]["c"] * scale), int(b["minimum"]["r"] * scale))
        p1 = (int(b["maximum"]["c"] * scale), int(b["maximum"]["r"] * scale))
        cv2.rectangle(v, p0, p1, COL.get(o["category"], INF), 2 if o["category"] in COL else 3)
    return v


if __name__ == "__main__":
    d = json.load(open(os.path.join(ROOT, "training.json")))
    rng = random.Random(7)
    sample = rng.sample(d, 20)
    rows = []
    tiles = []
    for k, rec in enumerate(sample):
        img = load_bgr(os.path.join(ROOT, rec["image"]["pathname"].lstrip("/")))
        boxes = [o["bounding_box"] for o in rec["objects"]]
        det = detect_cells(img)
        inv = 1 / det["scale"]
        centres = [((c["bbox"][0] + c["bbox"][2] / 2) * inv, (c["bbox"][1] + c["bbox"][3] / 2) * inv, c) for c in det["cells"] if c["kept"]]

        def inside(x, y):
            return any(b["minimum"]["c"] <= x <= b["maximum"]["c"] and b["minimum"]["r"] <= y <= b["maximum"]["r"] for b in boxes)

        unboxed = [(x, y) for x, y, c in centres if not inside(x, y)]
        # GT boxes (excluding leukocyte/difficult) containing no detected centre
        rbc = [b for o, b in zip(rec["objects"], boxes) if o["category"] not in ("leukocyte", "difficult")]
        missed = sum(1 for b in rbc if not any(b["minimum"]["c"] <= x <= b["maximum"]["c"] and b["minimum"]["r"] <= y <= b["maximum"]["r"] for x, y, _ in centres))
        rows.append((len(boxes), len(centres), len(unboxed), missed))
        if k < 12:
            v = draw(img, rec["objects"], 0.42)
            for x, y in unboxed:
                cv2.circle(v, (int(x * 0.42), int(y * 0.42)), 9, (255, 0, 255), 2)   # magenta = detected cell with NO GT box
            cv2.putText(v, f"#{k} boxes={len(boxes)} detected={len(centres)} unboxed={len(unboxed)}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            tiles.append(v)
        print(f"img{k:2d}: GT boxes={len(boxes):3d}  detected cells={len(centres):3d}  detected-but-unboxed={len(unboxed):3d}  GT-cells-missed-by-detector={missed:3d}", flush=True)
    r = np.array(rows)
    print(f"\nTOTAL over 20 images: GT boxes {r[:,0].sum()}, detected {r[:,1].sum()}, detected-but-unboxed {r[:,2].sum()} "
          f"({r[:,2].sum() / max(r[:,1].sum(), 1):.1%} of detections), GT cells missed by detector {r[:,3].sum()}")
    for name, grp in (("A", tiles[:6]), ("B", tiles[6:12])):
        h = min(t.shape[0] for t in grp); grp = [t[:h] for t in grp]
        cv2.imwrite(os.path.join(OUT, f"complete_{name}.png"), np.concatenate([np.concatenate(grp[:3], 1), np.concatenate(grp[3:], 1)], 0))
