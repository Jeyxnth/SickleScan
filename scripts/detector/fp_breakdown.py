"""
(1) What are the classifier>=0.5 detections on BBBC041 VAL malaria-negative fields? (matched labelled RBC / leukocyte / unmatched)
(2) Same-domain sensitivity: BBBC041 VAL fields that DO contain infected cells, same image-level rules, so a rule can be
    judged on false positives and sensitivity from the same held-out split. Usage: python scripts/detector/fp_breakdown.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join("scripts", "bbbc041"))
sys.path.insert(0, os.path.dirname(__file__))
import pipeline as P
from common import INFECTED, gt_boxes, image_path, iou_matrix, read_bgr, splits
from fp_bbbc_negatives import negatives

THR = 0.2  # score floor; higher thresholds are subsets

if __name__ == "__main__":
    det, cls = P.load_models()
    sp = splits()
    neg = negatives(sp["val"])
    pos = [r for r in sp["val"] if any(o["category"] in INFECTED for o in r["objects"])]
    kinds = {"labelled RBC": 0, "leukocyte": 0, "no labelled cell (IoU<0.3)": 0}
    top = []
    for r in neg:
        b, sc, p = P.run(det, cls, read_bgr(image_path(r)), THR)
        gb, inf, cell = gt_boxes(r)
        cats = np.array([o["category"] for o in r["objects"]])
        iou = iou_matrix(b, gb)
        for i in np.where(p >= .5)[0]:
            j = iou[i].argmax() if iou.shape[1] else -1
            if j >= 0 and iou[i, j] >= 0.3:
                kinds["leukocyte" if cats[j] == "leukocyte" else "labelled RBC"] += 1
            else:
                kinds["no labelled cell (IoU<0.3)"] += 1
        top.append(np.round(np.sort(p[p >= .5])[::-1][:5], 2).tolist())
    print("Classifier>=0.5 detections on the 30 negative val fields, by what they overlap:", kinds)
    print("scores of flagged cells per flagged field:", [t for t in top if t])

    print(f"\nSame-domain sensitivity: {len(pos)} BBBC041 val fields containing infected cells (held out)")
    P_ = []
    for r in pos:
        b, sc, p = P.run(det, cls, read_bgr(image_path(r)), THR)
        P_.append((sc, p))
    N_ = []
    for r in neg:
        b, sc, p = P.run(det, cls, read_bgr(image_path(r)), THR)
        N_.append((sc, p))
    rules = {"any cell >= 0.5": lambda q: (q >= .5).any(), "any cell >= 0.9": lambda q: (q >= .9).any(),
             ">=2 cells >= 0.5": lambda q: (q >= .5).sum() >= 2, ">=3 cells >= 0.5": lambda q: (q >= .5).sum() >= 3,
             ">=1 cell >= 0.9 or >=3 >= 0.5": lambda q: (q >= .9).any() or (q >= .5).sum() >= 3}
    for thr in (0.2, 0.3, 0.4):
        print(f"detector thr {thr}:")
        for name, fn in rules.items():
            s = sum(bool(fn(p[sc >= thr])) for sc, p in P_)
            f = sum(bool(fn(p[sc >= thr])) for sc, p in N_)
            print(f"   rule '{name}': positive fields caught {s}/{len(pos)} = {s / len(pos):.0%} | negative fields flagged {f}/{len(neg)} = {f / len(neg):.0%}")
