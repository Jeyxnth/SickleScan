"""
Phase 14: guardrail v2 manifest = the existing Phase 7 manifest (same rows, same splits, so old and new results are
comparable) + BBBC041 wide-field images as extra "smear" examples.

BBBC041 rows come ONLY from the Phase 9b training split (1,027 images; seed-42 85/15 image split of training.json). They are divided
80/20 into guardrail-train / guardrail-val (for early stopping). The 181 held-out validation images (which contain the 163 fields
used in Phases 11-13) and the official test.json (120 images, a different microscope setup) are never added to training; they are
written to a separate evaluation list.
Writes data/guardrail/manifest_v2.csv (train/val/test as before) and data/guardrail/bbbc_eval.csv (held-out evaluation only).
Usage: python scripts/guardrail_v2/build_manifest_v2.py
"""
import csv
import os
import random
import sys

sys.path.insert(0, os.path.join("scripts", "detector"))
from common import INFECTED, image_path, splits

OLD = os.path.join("data", "guardrail", "manifest.csv")
NEW = os.path.join("data", "guardrail", "manifest_v2.csv")
EVAL = os.path.join("data", "guardrail", "bbbc_eval.csv")


def rel(rec):
    return image_path(rec).replace("\\", "/")


if __name__ == "__main__":
    sp = splits()
    rows = list(csv.DictReader(open(OLD, newline="", encoding="utf-8")))
    train_imgs = sorted(sp["train"], key=lambda r: r["image"]["pathname"])
    rng = random.Random(42)
    rng.shuffle(train_imgs)
    n_val = int(round(0.2 * len(train_imgs)))
    g_val, g_train = train_imgs[:n_val], train_imgs[n_val:]
    added = [{"path": rel(r), "label": "smear", "source": "bbbc041_field", "split": "train"} for r in g_train] + \
            [{"path": rel(r), "label": "smear", "source": "bbbc041_field", "split": "val"} for r in g_val]
    # hard guarantees: nothing held out can be in training
    held = {rel(r) for r in sp["val"]} | {rel(r) for r in sp["test"]}
    assert not ({a["path"] for a in added} & held)
    with open(NEW, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "source", "split"])
        w.writeheader()
        w.writerows(rows + added)
    # evaluation list: the 163 validation fields used in Phase 11-13 (>=1 infected box, or no infected / no difficult) + the 120 official test images
    neg_ok = lambda r: not any(o["category"] in INFECTED or o["category"] == "difficult" for o in r["objects"])
    pos_ok = lambda r: any(o["category"] in INFECTED for o in r["objects"])
    val163 = [r for r in sp["val"] if pos_ok(r) or neg_ok(r)]
    with open(EVAL, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "source", "split"])
        w.writeheader()
        for r in val163:
            w.writerow({"path": rel(r), "label": "smear", "source": "bbbc041_val163", "split": "eval"})
        for r in sp["test"]:
            w.writerow({"path": rel(r), "label": "smear", "source": "bbbc041_test120", "split": "eval"})
    print(f"BBBC041 rows added: guardrail-train {len(g_train)}, guardrail-val {len(g_val)} (from the {len(train_imgs)} training-split images)")
    print(f"held-out evaluation only: {len(val163)} validation fields (expect 163), {len(sp['test'])} official test images")
