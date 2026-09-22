"""Phase 14d: identify exactly which images flipped decision on the phone (new resize) vs TensorFlow reference, and whether they
fall in a reported test/eval split, so we can say precisely whether any published number is affected.
Usage: python scripts/device/identify_flips.py"""
import csv
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join("scripts", "guardrail"))
import data_pipeline as dp

P = os.path.join("demo_images", "device_checks", "parity")
ASSETS = os.path.join("android", "app", "src", "main", "assets")


def prob(model_file, x):
    it = tf.lite.Interpreter(model_path=os.path.join(ASSETS, model_file)); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(i["index"], x[None].astype(np.float32)); it.invoke()
    return float(it.get_tensor(o["index"]).reshape(-1)[0])


def status(p, ceiling=65.0):
    conf = (p if p >= .5 else 1 - p) * 100
    return "borderline" if conf < ceiling else ("positive" if p >= .5 else "negative")


if __name__ == "__main__":
    sets = {(r, m): s for r, m, s in (l.rstrip("\n").split(",", 2) for l in open(os.path.join(P, "sets.csv")))}
    res = {(r["path"], r["model"]): float(r["P"]) for r in csv.DictReader(open(os.path.join(P, "results.csv")))}

    print("=== sickle model on all 569 sickle images: which image(s) flip / change status, and are they in the reported 86-image test set? ===")
    split_of = {}
    for split in ("train", "val", "test"):
        for row in csv.DictReader(open(os.path.join("data", "splits", f"{split}.csv"), newline="", encoding="utf-8")):
            split_of[row["path"].replace("\\", "/")] = split
    print(f"loaded data/splits/{{train,val,test}}.csv: {len(split_of)} images "
          f"(train {sum(v == 'train' for v in split_of.values())}, val {sum(v == 'val' for v in split_of.values())}, test {sum(v == 'test' for v in split_of.values())})")
    import glob
    labelled = ([(p, "positive") for p in sorted(glob.glob("data/raw/Positive/Unlabelled/*.jpg"))]
                + [(p, "negative") for p in sorted(glob.glob("data/raw/Negative/Clear/*.jpg"))])
    sk = [p for p, _ in labelled]
    rel = {p: f"sickle/{lbl}__{os.path.basename(p)}" for p, lbl in labelled}
    flips, changes = [], []
    for p in sk:
        r = rel[p]
        pd_ = res.get((r, "sickle"))
        if pd_ is None:
            continue
        pt = prob("sicklescan_model.tflite", dp.load_and_preprocess_single(p))
        sp = split_of.get(p.replace("\\", "/"), "UNKNOWN")
        if (pd_ >= .5) != (pt >= .5):
            flips.append((p, pt, pd_, sp))
        if status(pt) != status(pd_):
            changes.append((p, pt, pd_, status(pt), status(pd_), sp))
    print(f"decision flips (n={len(flips)}):")
    for p, pt, pd_, sp in flips:
        print(f"  [{sp}] {p}: TensorFlow {pt:.4f} -> phone(new) {pd_:.4f}")
    print(f"status changes (n={len(changes)}):")
    for p, pt, pd_, st, sd, sp in changes:
        print(f"  [{sp}] {p}: TensorFlow {pt:.4f} ({st}) -> phone(new) {pd_:.4f} ({sd})")
    print(f"\nany flip or status change landing in the TEST split (the reported 86-image accuracy/sensitivity/specificity): "
          f"{'YES -- reported numbers would change' if any(f[3] == 'test' for f in flips) or any(c[5] == 'test' for c in changes) else 'NO'}")

    print("\n=== guardrail on the 120 official BBBC test images: which image flips? ===")
    ev = list(csv.DictReader(open("data/guardrail/bbbc_eval.csv", newline="", encoding="utf-8")))
    for r in ev:
        if r["source"] != "bbbc041_test120":
            continue
        rel_ = "bbbc/" + os.path.basename(r["path"])
        pd_ = res.get((rel_, "guardrail"))
        if pd_ is None:
            continue
        pt = prob("guardrail_model.tflite", dp.load_and_preprocess_single(r["path"]))
        if (pd_ >= .5) != (pt >= .5):
            print(f"  FLIP: {r['path']}: TensorFlow {pt:.4f} -> phone(new) {pd_:.4f}  (this changes the 'accepted' count for this set by 1)")
