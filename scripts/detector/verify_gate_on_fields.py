"""
Phase 13 gate check: does the minimum-cell-count gate (fewer than MIN_CELLS detected cells -> Inconclusive) change any of the 163
validated BBBC041 field results? Uses the deployed float16 pipeline's per-field cell counts and top scores
(artifacts/detector/tflite_pipeline_scores_float16.npz, from verify_tflite_pipeline.py) and the shipped rule (top cell >= 0.985).
Compares, per field, the answer without the gate (Positive / Negative) and with it (Positive / Negative / Inconclusive).
Usage: python scripts/detector/verify_gate_on_fields.py
"""
import os

import numpy as np

MIN_CELLS, T = 20, 0.985
z = np.load(os.path.join("artifacts", "detector", "tflite_pipeline_scores_float16.npz"))
mx, ncell, is_neg = z["mx"], z["ncell"], z["is_neg"]


def answer(cells, top, gated):
    if gated and cells < MIN_CELLS:
        return "Inconclusive"
    if cells == 0:
        return "Inconclusive"
    return "Positive" if top >= T else "Negative"


before = [answer(c, t, False) for c, t in zip(ncell, mx)]
after = [answer(c, t, True) for c, t in zip(ncell, mx)]
changed = [i for i in range(len(mx)) if before[i] != after[i]]
flag_changed = [i for i in range(len(mx)) if (before[i] == "Positive") != (after[i] == "Positive")]
print(f"{len(mx)} validation fields ({int((~is_neg).sum())} infected, {int(is_neg.sum())} negative), gate: fewer than {MIN_CELLS} cells -> Inconclusive")
print(f"fields with fewer than {MIN_CELLS} detected cells: {int((ncell < MIN_CELLS).sum())} -> {[(('negative' if is_neg[i] else 'infected'), int(ncell[i]), round(float(mx[i]), 3)) for i in np.where(ncell < MIN_CELLS)[0]]}")
print(f"Positive/not-Positive decision changed for {len(flag_changed)}/{len(mx)} fields")
print(f"displayed answer changed for {len(changed)}/{len(mx)} fields: {[(('negative' if is_neg[i] else 'infected'), before[i], '->', after[i], int(ncell[i])) for i in changed]}")
for name, arr in (("without gate", before), ("with gate", after)):
    inf, neg = np.array(arr)[~is_neg], np.array(arr)[is_neg]
    print(f"  {name:12s}: infected fields Positive {int((inf == 'Positive').sum())}/{len(inf)}, Inconclusive {int((inf == 'Inconclusive').sum())} | "
          f"negative fields Positive {int((neg == 'Positive').sum())}/{len(neg)}, Negative {int((neg == 'Negative').sum())}, Inconclusive {int((neg == 'Inconclusive').sum())}")
