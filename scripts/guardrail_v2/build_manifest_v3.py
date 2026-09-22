"""
Phase 14b: guardrail v3 manifest = v2 manifest (Phase 7 rows + 822/205 BBBC041 wide-field images) + BBBC041 SINGLE-CELL crops as extra smear examples.
Crops come ONLY from artifacts/bbbc041/crops_train.npz (built from the Phase 9b training-split images). 800 go to guardrail-train (400 infected +
400 uninfected RBC), 200 to guardrail-val, assigned by the SOURCE IMAGE's split in the v2 manifest (so a crop can never be in guardrail-train while
its field image is in guardrail-val). crops_val (held-out images) and crops_test (official test set) are never added. Crops are written as PNG to
data/guardrail/bbbc_crops/ (local, gitignored). Writes data/guardrail/manifest_v3.csv.
Usage: python scripts/guardrail_v2/build_manifest_v3.py
"""
import csv
import os
import random

import cv2
import numpy as np

V2 = os.path.join("data", "guardrail", "manifest_v2.csv")
V3 = os.path.join("data", "guardrail", "manifest_v3.csv")
OUT = os.path.join("data", "guardrail", "bbbc_crops")

if __name__ == "__main__":
    rows = list(csv.DictReader(open(V2, newline="", encoding="utf-8")))
    img_split = {os.path.basename(r["path"]): r["split"] for r in rows if r["source"] == "bbbc041_field"}
    z = np.load(os.path.join("artifacts", "bbbc041", "crops_train.npz"), allow_pickle=True)
    X, y, imgs = z["X"], z["y"], z["imgs"]
    rng = random.Random(42)
    added, counts = [], {"train": 0, "val": 0}
    for split, want in (("train", 400), ("val", 100)):
        for label_name, is_inf in (("infected", 1), ("uninfected", 0)):
            idx = [i for i in range(len(y)) if int(y[i]) == is_inf and img_split.get(str(imgs[i])) == split]
            for i in rng.sample(idx, want):
                d = os.path.join(OUT, split); os.makedirs(d, exist_ok=True)
                p = os.path.join(d, f"crop_{label_name}_{i}.png").replace("\\", "/")
                cv2.imencode(".png", cv2.cvtColor(X[i], cv2.COLOR_RGB2BGR))[1].tofile(p)
                added.append({"path": p, "label": "smear", "source": "bbbc041_cell", "split": split})
                counts[split] += 1
    with open(V3, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "source", "split"])
        w.writeheader(); w.writerows(rows + added)
    print(f"single-cell crops added: guardrail-train {counts['train']}, guardrail-val {counts['val']}")
    import collections
    print(collections.Counter((r['label'], r['split']) for r in rows + added))
