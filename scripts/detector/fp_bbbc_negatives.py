"""
False-positive check of detector + BBBC041 classifier on genuine malaria-NEGATIVE BBBC041 full fields.
Negative = image with no infected-stage box and no 'difficult' box (annotator-labelled; unannotated infected cells
cannot be excluded). Sets: VAL (held out from training of both detector and classifier) and TEST (official, tiny).
Usage: python scripts/detector/fp_bbbc_negatives.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join("scripts", "bbbc041"))
sys.path.insert(0, os.path.dirname(__file__))
import pipeline as P
from common import INFECTED, gt_boxes, image_path, iou_matrix, read_bgr, splits


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def negatives(recs):
    return [r for r in recs if not any(o["category"] in INFECTED or o["category"] == "difficult" for o in r["objects"])]


if __name__ == "__main__":
    det, cls = P.load_models()
    sp = splits()
    for name in ("val", "test"):
        recs = negatives(sp[name])
        # run once at the lowest threshold; higher thresholds are subsets by detector score
        cache = []
        n_gt = 0
        for r in recs:
            img = read_bgr(image_path(r))
            b, sc, p = P.run(det, cls, img, 0.2)
            gb, inf, cell = gt_boxes(r)
            n_gt += int(cell.sum())
            cache.append((b, sc, p))
        print(f"\n===== BBBC041 {name.upper()} malaria-negative fields: {len(recs)} images, {n_gt} labelled cells =====")
        for thr in (0.2, 0.3, 0.4):
            det_n = flag = 0
            f_any = f_3 = f_5 = 0
            for b, sc, p in cache:
                k = sc >= thr
                q = p[k]
                det_n += len(q); flag += int((q >= .5).sum())
                f_any += bool((q >= .5).any()); f_3 += bool((q >= .5).sum() >= 3); f_5 += bool((q >= .9).sum() >= 5)
            lo, hi = wilson(flag, det_n)
            fl, fh = wilson(f_any, len(recs))
            print(f"detector thr {thr}: detected {det_n} cells ({det_n / max(n_gt, 1):.0%} of labelled) | per-cell >=0.5: {flag}/{det_n} = "
                  f"{flag / max(det_n, 1):.2%} [95% CI {lo:.2%}-{hi:.2%}] | fields flagged: any>=0.5 {f_any}/{len(recs)} "
                  f"[CI {fl:.0%}-{fh:.0%}], >=3 cells>=0.5 {f_3}/{len(recs)}, >=5 cells>=0.9 {f_5}/{len(recs)}")
