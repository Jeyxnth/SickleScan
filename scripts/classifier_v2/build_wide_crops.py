"""Step 1b: rebuild the train/val GT-box crops with 2.0x padding (more neighbouring cells visible) instead of 1.15x.
Identical splits/sampling/seed to scripts/bbbc041/build_crops.py (test split not rebuilt); writes to a separate folder so
the baseline crops are untouched. Usage: python scripts/classifier_v2/build_wide_crops.py"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join("scripts", "bbbc041"))
import build_crops as B

PAD = 2.0
OUT = os.path.join("artifacts", "classifier_v2", "wide")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    B.OUT, B.PAD = OUT, PAD
    rng = random.Random(42)
    train_all = json.load(open(os.path.join(B.ROOT, "training.json")))
    idx = list(range(len(train_all)))
    rng.shuffle(idx)
    n_val = int(round(0.15 * len(idx)))
    val = [train_all[i] for i in idx[:n_val]]
    train = [train_all[i] for i in idx[n_val:]]
    B.build("train", train, rng)
    B.build("val", val, rng)
