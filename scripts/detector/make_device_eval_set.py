"""
Phase 13: build a small on-device evaluation set from BBBC041 VALIDATION fields (never the reserved photos): copies
N infected + M negative fields to demo_images/wide_field_eval/ (gitignored -- BBBC041 is CC BY-NC-SA, not redistributed
in the repo) and writes expected.csv from the Python TFLite pipeline (float16 detector + bundled classifier), so the
instrumented test (WideFieldPipelineInstrumentedTest) can check the phone against it and report real timings.
Usage: python scripts/detector/make_device_eval_set.py [N_INFECTED] [N_NEGATIVE]
"""
import os
import random
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join("scripts", "bbbc041"))
sys.path.insert(0, os.path.join("scripts", "detector"))
sys.path.insert(0, os.path.join("scripts", "classifier_v2"))
from common import INFECTED, image_path, splits
from fp_bbbc_negatives import negatives

if __name__ == "__main__":
    n_inf = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n_neg = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    sp = splits()["val"]
    neg_ids = {r["image"]["pathname"] for r in negatives(sp)}
    pos_ids = {r["image"]["pathname"] for r in sp if any(o["category"] in INFECTED for o in r["objects"])}
    recs = [r for r in sp if r["image"]["pathname"] in neg_ids | pos_ids]        # same order as the saved score arrays
    z = np.load(os.path.join("artifacts", "detector", "tflite_pipeline_scores_float16.npz"))
    mx, ncell, is_neg = z["mx"], z["ncell"], z["is_neg"]
    assert len(recs) == len(mx)
    rng = random.Random(7)
    pos_idx = [i for i in range(len(recs)) if not is_neg[i]]
    neg_idx = [i for i in range(len(recs)) if is_neg[i]]
    pick = rng.sample(pos_idx, n_inf) + rng.sample(neg_idx, n_neg)
    out = os.path.join("demo_images", "wide_field_eval")
    os.makedirs(out, exist_ok=True)
    rows = ["file,field_label,cells,top_score"]
    for i in pick:
        src = image_path(recs[i])
        name = os.path.basename(src)
        shutil.copy(src, os.path.join(out, name))
        rows.append(f"{name},{'negative' if is_neg[i] else 'infected'},{int(ncell[i])},{mx[i]:.6f}")
    open(os.path.join(out, "expected.csv"), "w").write("\n".join(rows) + "\n")
    print(f"{len(pick)} fields -> {out}; total {sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1e6:.0f} MB")
