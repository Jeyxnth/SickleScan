"""
Task 3: run the four user photos through the BBBC041 pipeline, plus controls.
  (a) heuristic detector (Phase 9a) -> raw native-scale crops -> BBBC041 classifier   [the full pipeline]
  (b) ORACLE crops for photo 2 (hand-marked by eye) -> classifier                     [isolates classifier from detector]
  (c) sickle-dataset fields (presumed malaria-free) as a false-positive control for the same pipeline
Hand-marked boxes are my visual read of photo 2, not lab ground truth.
"""
import glob
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join("scripts", "tiling"))
sys.path.insert(0, os.path.dirname(__file__))
from build_crops import crop
from classify_tiles import make_crops
from detect_cells import detect_cells, load_bgr

OUT = os.path.join("artifacts", "bbbc041")
PHOTOS = [os.path.join("data", "tiling_validation", "malaria_positive_fields", f) for f in ("1.webp", "2.jpg", "3.jpg", "4.jpg")]

# photo 2, (x0, y0, x1, y1) in original pixels -- by eye
P2_INFECTED = {"schizont": (228, 105, 275, 148), "ring/troph": (88, 108, 138, 146), "infected@centre": (175, 187, 214, 224)}
P2_HEALTHY = [(193, 25, 234, 67), (240, 23, 278, 63), (292, 193, 330, 230), (200, 218, 245, 250), (53, 60, 90, 95),
              (105, 225, 137, 255), (357, 162, 395, 193), (343, 102, 380, 140), (257, 67, 295, 97), (28, 250, 63, 285)]


def predict(model, crops_rgb):
    return model.predict((np.stack(crops_rgb).astype(np.float32) - 127.5) / 127.5, batch_size=64, verbose=0).reshape(-1)


def box(b):
    return {"minimum": {"r": b[1], "c": b[0]}, "maximum": {"r": b[3], "c": b[2]}}


def montage(crops, probs, path, k=12, title=""):
    idx = np.argsort(-np.asarray(probs))[:k]
    tiles = []
    for i in idx:
        t = cv2.resize(cv2.cvtColor(crops[i], cv2.COLOR_RGB2BGR), (112, 112))
        cv2.putText(t, f"{probs[i]:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % 6:
        tiles.append(np.zeros((112, 112, 3), np.uint8))
    rows = [np.concatenate(tiles[r:r + 6], 1) for r in range(0, len(tiles), 6)]
    m = np.concatenate(rows, 0)
    bar = np.zeros((20, m.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(path, np.concatenate([bar, m], 0))


def summarize(p):
    s = np.sort(p)[::-1]
    return f"n={len(p):3d} >=0.5:{int((p >= .5).sum()):3d} >=0.9:{int((p >= .9).sum()):3d} max={s[0]:.3f} top3={s[:3].mean():.3f}"


if __name__ == "__main__":
    model = tf.keras.models.load_model(os.path.join(OUT, "model", "bbbc_keras.keras"))
    old = {os.path.basename(r["path"].replace("\\", "/")): r["whole"] for r in __import__("json").load(open("artifacts/tiling/tiling_eval.json"))["extra"]}

    print("=== (a) FULL PIPELINE: heuristic detector -> BBBC041 classifier ===")
    overlays = []
    for f in PHOTOS:
        img = load_bgr(f)
        det = detect_cells(img)
        cells, raw, _ = make_crops(det, img)
        p = predict(model, raw) if raw else np.zeros(0)
        name = os.path.basename(f)
        print(f"{name:8s} old whole-image(NIH model)={old.get(name, float('nan')):.3f} | new: {summarize(p)}")
        if len(p):
            montage(raw, p, os.path.join(OUT, f"top_{name.split('.')[0]}.png"), title=f"photo {name}: top-12 crops by P(infected)")
        vis = cv2.resize(img, (500, 500), interpolation=cv2.INTER_CUBIC)
        sx, sy = 500 / img.shape[1], 500 / img.shape[0]
        for c, q in zip(cells, p):
            x, y, bw, bh = c["bbox"]
            inv = 1 / det["scale"]
            cv2.rectangle(vis, (int(x * inv * sx), int(y * inv * sy)), (int((x + bw) * inv * sx), int((y + bh) * inv * sy)),
                          (0, 0, 255) if q >= 0.5 else (0, 200, 0), 1)
        cv2.putText(vis, name, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        overlays.append(vis)
    cv2.imwrite(os.path.join(OUT, "photo_overlays_new.png"), np.concatenate(overlays, 1))

    print("\n=== (b) ORACLE crops on photo 2 (hand-marked; classifier only, no detector) ===")
    img2 = load_bgr(PHOTOS[1])
    inf_crops = [crop(img2, box(b)) for b in P2_INFECTED.values()]
    healthy_crops = [crop(img2, box(b)) for b in P2_HEALTHY]
    pi, ph = predict(model, inf_crops), predict(model, healthy_crops)
    for (k, b), q in zip(P2_INFECTED.items(), pi):
        print(f"  infected  {k:16s} P(infected)={q:.3f}")
    print(f"  healthy   n={len(ph)}: P(infected) min {ph.min():.3f} median {np.median(ph):.3f} max {ph.max():.3f}; >=0.5: {(ph >= .5).sum()}")
    montage(inf_crops + healthy_crops, np.concatenate([pi, ph]), os.path.join(OUT, "photo2_oracle.png"), k=13,
            title="photo 2 oracle crops (first 3 = infected by eye), sorted by P(infected)")

    print("\n=== (c) CONTROLS: sickle-dataset fields (presumed malaria-free) through the SAME pipeline ===")
    files = sorted(glob.glob("data/raw/Negative/Clear/*.jpg"))[:20] + sorted(glob.glob("data/raw/Positive/Unlabelled/*.jpg"))[:10]
    allp, per_field = [], []
    for f in files:
        img = load_bgr(f)
        det = detect_cells(img)
        cells, raw, _ = make_crops(det, img)
        p = predict(model, raw) if raw else np.zeros(0)
        per_field.append(p)
        allp.append(p)
    allp = np.concatenate(allp)
    print(f"{len(files)} fields, {len(allp)} cells: per-cell P>=0.5 {np.mean(allp >= .5):.2%}, >=0.9 {np.mean(allp >= .9):.2%}, >=0.99 {np.mean(allp >= .99):.2%}")
    for name, fn in (("any cell >= 0.5", lambda q: (q >= .5).any()), ("any cell >= 0.9", lambda q: (q >= .9).any()),
                     (">=3 cells >= 0.5", lambda q: (q >= .5).sum() >= 3), (">=5 cells >= 0.9", lambda q: (q >= .9).sum() >= 5),
                     (">=10% of cells >= 0.5", lambda q: len(q) > 0 and (q >= .5).mean() >= .10)):
        print(f"  rule '{name}': flags {sum(fn(q) for q in per_field)}/{len(per_field)} malaria-free fields")
