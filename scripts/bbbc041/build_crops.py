"""
Build native-scale cell crops from BBBC041 ground-truth boxes.

Labels: infected = trophozoite / schizont / gametocyte / ring (all P. vivax stages, treated as one class);
        not infected = red blood cell + leukocyte (WBCs kept as hard negatives -- they were a false-positive
        source in Phase 9a). "difficult" boxes are excluded (ambiguous ground truth).
Splits are BY IMAGE (never crop-level, so cells from one photo can't leak across splits):
        train / val = 85% / 15% of training.json images (seed 42); test = official test.json (120 images,
        a different resolution -- 1944x1383 vs 1600x1200 -- so it is also a mild domain-shift test).
Sampling: all infected + all leukocyte crops are kept; red-cell crops are randomly subsampled
        (train 6000, val 1500; test keeps everything) to keep the ~2.7% infected prevalence from swamping
        training. Crops are square (box side x 1.15, centred), edge-replicated at image borders, resized to 224.
"""
import json
import os
import random

import cv2
import numpy as np

ROOT = os.path.join("data", "bbbc041", "raw", "malaria")
OUT = os.path.join("artifacts", "bbbc041")
os.makedirs(OUT, exist_ok=True)
INFECTED = {"trophozoite", "schizont", "gametocyte", "ring"}
PAD = 1.15
N_RBC = {"train": 6000, "val": 1500, "test": None}


def crop(img, b):
    r0, c0, r1, c1 = b["minimum"]["r"], b["minimum"]["c"], b["maximum"]["r"], b["maximum"]["c"]
    side = int(round(max(r1 - r0, c1 - c0) * PAD))
    side = max(side, 16)
    cy, cx = (r0 + r1) / 2.0, (c0 + c1) / 2.0
    y0, x0 = int(round(cy - side / 2)), int(round(cx - side / 2))
    H, W = img.shape[:2]
    top, left = max(0, -y0), max(0, -x0)
    bottom, right = max(0, y0 + side - H), max(0, x0 + side - W)
    if top or left or bottom or right:
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
        y0, x0 = y0 + top, x0 + left
    patch = img[y0:y0 + side, x0:x0 + side]
    return cv2.cvtColor(cv2.resize(patch, (224, 224), interpolation=cv2.INTER_AREA if side > 224 else cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)


def build(split_name, records, rng):
    items = []
    for rec in records:
        for o in rec["objects"]:
            if o["category"] == "difficult":
                continue
            items.append((rec, o))
    keep = [it for it in items if it[1]["category"] != "red blood cell"]
    rbc = [it for it in items if it[1]["category"] == "red blood cell"]
    n = N_RBC[split_name]
    if n is not None and len(rbc) > n:
        rbc = rng.sample(rbc, n)
    items = keep + rbc
    items.sort(key=lambda it: it[0]["image"]["pathname"])
    X, y, cats, imgs, boxes = [], [], [], [], []
    cache_path, cache = None, None
    for rec, o in items:
        p = os.path.join(ROOT, rec["image"]["pathname"].lstrip("/"))
        if p != cache_path:
            cache = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            cache_path = p
        X.append(crop(cache, o["bounding_box"]))
        y.append(1 if o["category"] in INFECTED else 0)
        cats.append(o["category"])
        imgs.append(os.path.basename(p))
        b = o["bounding_box"]
        boxes.append([b["minimum"]["r"], b["minimum"]["c"], b["maximum"]["r"], b["maximum"]["c"]])
    np.savez(os.path.join(OUT, f"crops_{split_name}.npz"), X=np.stack(X), y=np.array(y, np.float32),
             cats=np.array(cats), imgs=np.array(imgs), boxes=np.array(boxes))
    from collections import Counter
    print(f"{split_name}: {len(X)} crops from {len({r['image']['pathname'] for r, _ in items})} images | "
          f"infected {sum(y)} ({np.mean(y):.1%}) | {dict(Counter(cats))}", flush=True)


if __name__ == "__main__":
    rng = random.Random(42)
    train_all = json.load(open(os.path.join(ROOT, "training.json")))
    test = json.load(open(os.path.join(ROOT, "test.json")))
    idx = list(range(len(train_all)))
    rng.shuffle(idx)
    n_val = int(round(0.15 * len(idx)))
    val = [train_all[i] for i in idx[:n_val]]
    train = [train_all[i] for i in idx[n_val:]]
    assert not ({r["image"]["pathname"] for r in val} & {r["image"]["pathname"] for r in train})
    build("train", train, rng)
    build("val", val, rng)
    build("test", test, rng)
