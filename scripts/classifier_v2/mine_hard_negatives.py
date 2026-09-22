"""
Step 1a: mine hard negatives for the crop classifier from the DETECTOR's own detections on TRAIN images only
(val/test untouched). Runs detector (thr 0.2) + the baseline classifier over each training image and keeps as
label-0 crops (detector-produced boxes, i.e. the box distribution seen at deployment):
  hard_rbc         detections overlapping a labelled RBC (IoU>=0.3) that the baseline scores >= 0.2
  unmatched_hard   detections overlapping no labelled cell (border fragments, WBC-like cells, debris), baseline >= 0.2
  unmatched_random a random sample (cap below) of the remaining unmatched detections with baseline < 0.2
Detections overlapping any infected/difficult ground-truth box (IoU>=0.3) are never used as negatives.
Caveat: the detector and baseline were trained on these images, so they err less here than on unseen fields; the
mined set underestimates deployment hard cases.
Usage: python scripts/classifier_v2/mine_hard_negatives.py
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join("scripts", "bbbc041"))
sys.path.insert(0, os.path.join("scripts", "detector"))
import pipeline as P
from build_crops import crop
from common import gt_boxes, image_path, iou_matrix, read_bgr, splits

OUT = os.path.join("artifacts", "classifier_v2")
CAP_RANDOM_UNMATCHED = 2500
MIN_SCORE = 0.2


def make_crop(img, bb):
    return crop(img, {"minimum": {"r": bb[1], "c": bb[0]}, "maximum": {"r": bb[3], "c": bb[2]}})


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    det, cls = P.load_models()
    rng = random.Random(0)
    train = splits()["train"]
    picked = []          # (image index, box, kind, baseline score)
    random_pool = []     # (image index, box, baseline score)
    stats = {"detections": 0, "near_infected_skipped": 0, "hard_rbc": 0, "unmatched_hard": 0, "unmatched_pool": 0}
    for n, r in enumerate(train):
        img = read_bgr(image_path(r))
        b, sc, p = P.run(det, cls, img, 0.2)
        gb, inf, cell = gt_boxes(r)
        cats = np.array([o["category"] for o in r["objects"]])
        near_inf = inf | (cats == "difficult")
        rbc = cell & ~near_inf
        iou = iou_matrix(b, gb)
        for i in range(len(b)):
            stats["detections"] += 1
            if iou.shape[1] and (iou[i][near_inf] >= 0.3).any():
                stats["near_infected_skipped"] += 1
            elif iou.shape[1] and (iou[i][rbc] >= 0.3).any():
                if p[i] >= MIN_SCORE:
                    picked.append((n, b[i], "hard_rbc", p[i])); stats["hard_rbc"] += 1
            elif p[i] >= MIN_SCORE:
                picked.append((n, b[i], "unmatched_hard", p[i])); stats["unmatched_hard"] += 1
            else:
                random_pool.append((n, b[i], p[i])); stats["unmatched_pool"] += 1
        if n % 100 == 0:
            print(n, stats, flush=True)
    rng.shuffle(random_pool)
    picked += [(n, bb, "unmatched_random", s) for n, bb, s in random_pool[:CAP_RANDOM_UNMATCHED]]
    picked.sort(key=lambda t: t[0])
    crops, kinds, scores, cache_n, img = [], [], [], -1, None
    for n, bb, kind, s in picked:
        if n != cache_n:
            img, cache_n = read_bgr(image_path(train[n])), n
        crops.append(make_crop(img, bb)); kinds.append(kind); scores.append(s)
    np.savez(os.path.join(OUT, "hard_negatives_train.npz"), X=np.stack(crops), y=np.zeros(len(crops), np.float32),
             kinds=np.array(kinds), baseline_score=np.array(scores, np.float32))
    from collections import Counter
    print("FINAL", stats, "| saved crops", len(crops), dict(Counter(kinds)), flush=True)
