"""
Phase 14: FRESH non-smear images never used to train or evaluate any guardrail (v1 or v2): new random samples (different seed) from the
same raw public sources, with every image that is a near-duplicate of an existing guardrail negative removed (16x16 grayscale
fingerprint comparison against all ~3,400 existing negatives). Written to data/guardrail/fresh_negatives/<source>/ (local, gitignored).
Sources: coco, places, dtd, lfw, screenshots (documents were used in full; synthetic failures are procedural).
Usage: python scripts/guardrail_v2/build_fresh_negatives.py
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join("scripts", "guardrail"))
import build_dataset as bd

FRESH = Path("data/guardrail/fresh_negatives")
WANT = {"coco": 250, "places": 250, "dtd": 200, "lfw": 200, "screenshots": 200}
POOL = {"coco": 450, "places": 320, "dtd": 300, "lfw": 300, "screenshots": 300}   # oversample, then drop duplicates
TOL = 6.0   # mean absolute difference (0-255) on the 16x16 fingerprint below which two images are treated as the same picture


def fp(path):
    im = Image.open(path).convert("L").resize((16, 16), Image.BILINEAR)
    return np.asarray(im, np.float32).reshape(-1)


if __name__ == "__main__":
    existing = [p for p in Path("data/guardrail/negatives").rglob("*.jpg") if not p.parent.name.startswith("synthetic")]
    E = np.stack([fp(p) for p in existing])
    print(f"{len(existing)} existing negatives fingerprinted", flush=True)
    bd.OUT = FRESH
    rng = random.Random(20260922)
    kept_total = 0
    for source, n in POOL.items():
        # coco: a fresh sample from the whole pool; the extractor writes thumbnails to FRESH/<source>/
        paths = bd.EXTRACTORS[source](rng, n)
        kept = 0
        for p in paths:
            f = fp(p)
            if np.abs(E - f).mean(axis=1).min() < TOL or kept >= WANT[source]:
                os.remove(p)
            else:
                kept += 1
        print(f"{source}: kept {kept} of {len(paths)} (duplicates of existing negatives / surplus removed)", flush=True)
        kept_total += kept
    print("fresh non-smear images:", kept_total)
