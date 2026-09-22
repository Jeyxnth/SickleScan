"""Would a minimum-cell-count gate ("fewer than k detected cells -> Inconclusive") separate real smear fields from non-smear photos?
Real fields: the 163 BBBC041 validation fields (cells per field from the deployed float16 pipeline). Non-smear: the 360 natural-image
results of nonsmear_test.py (coco, documents, dtd, lfw, places, screenshots; the 240 synthetic images all already give zero cells).
Analysis only -- nothing is shipped or tuned on anything but these numbers, so treat it as a proposal to validate, not a result.
Usage: python scripts/detector/cellcount_gate_analysis.py"""
import csv
import os

import numpy as np

z = np.load(os.path.join("artifacts", "detector", "tflite_pipeline_scores_float16.npz"))
ncell, is_neg, mx = z["ncell"], z["is_neg"], z["mx"]
rows = list(csv.DictReader(open(os.path.join("model_output", "detector", "phase13_nonsmear_per_image.csv"))))
nat = [r for r in rows if not r["category"].startswith("synthetic")]
cells = np.array([int(r["cells"]) for r in nat]); top = np.array([float(r["top_score"]) for r in nat])
cats = np.array([r["category"] for r in nat])
T = 0.985
print(f"BBBC041 validation fields: cells per field min {ncell.min()}, 5th percentile {np.percentile(ncell, 5):.0f}, median {np.median(ncell):.0f}; "
      f"infected-only min {ncell[~is_neg].min()}, negative-only min {ncell[is_neg].min()}")
print(f"Non-smear natural images ({len(nat)}): cells per image median {np.median(cells):.0f}, 95th percentile {np.percentile(cells, 95):.0f}, max {cells.max()}\n")
print(f"{'min cells k':>11s} | BBBC infected fields kept & caught | BBBC negative fields flagged | non-smear: give a verdict (Neg or Pos) | non-smear: Positive")
for k in (1, 5, 10, 15, 20, 30, 40):
    pos_ok = ((ncell >= k) & (mx >= T) & ~is_neg).sum(); neg_flag = ((ncell >= k) & (mx >= T) & is_neg).sum()
    verdict = (cells >= k).sum(); posv = ((cells >= k) & (top >= T)).sum()
    print(f"{k:11d} | {pos_ok:3d}/{int((~is_neg).sum())} = {pos_ok / (~is_neg).sum():5.1%}                | {neg_flag:2d}/{int(is_neg.sum())} = {neg_flag / is_neg.sum():5.1%}"
          f"            | {verdict:3d}/{len(nat)} = {verdict / len(nat):5.1%}                      | {posv}/{len(nat)} = {posv / len(nat):.1%}")
print("\nby category, k=20:", {c: int(((cells >= 20) & (cats == c)).sum()) for c in sorted(set(cats))}, "of 60 each (documents 60)")
