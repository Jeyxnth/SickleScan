"""
Run detection + per-cell malaria scoring over field images and print per-image and aggregate stats.
Usage: python scripts/tiling/run_tiling_eval.py <n_clear> <n_sickle_pos> [extra_image_paths...]
Fields from the sickle dataset are NOT malaria-labelled; they are used as a false-positive baseline
(assumed malaria-free, which is not certified) and to gauge detector behaviour.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from classify_tiles import classify_field, load_model, summarize

OUT = os.path.join("artifacts", "tiling")


def montage(crops, probs, path, k=12):
    idx = np.argsort(-np.asarray(probs))[:k]
    tiles = []
    for i in idx:
        t = cv2.cvtColor(crops[i], cv2.COLOR_RGB2BGR)
        t = cv2.resize(t, (112, 112))
        cv2.putText(t, f"{probs[i]:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % 6:
        tiles.append(np.zeros((112, 112, 3), np.uint8))
    rows = [np.concatenate(tiles[r:r + 6], 1) for r in range(0, len(tiles), 6)]
    cv2.imwrite(path, np.concatenate(rows, 0))


if __name__ == "__main__":
    n_clear, n_sick = int(sys.argv[1]), int(sys.argv[2])
    extra = sys.argv[3:]
    model = load_model()
    groups = {
        "clear": sorted(glob.glob("data/raw/Negative/Clear/*.jpg"))[:n_clear],
        "sickle_pos": sorted(glob.glob("data/raw/Positive/Unlabelled/*.jpg"))[:n_sick],
        "extra": extra,
    }
    results = {}
    for g, files in groups.items():
        rows = []
        for f in files:
            r = classify_field(model, f)
            raw, masked, cells = r.pop("_crops")
            sr, sm = summarize(r["p_raw"]), summarize(r["p_masked"])
            rows.append(dict(path=f, blobs=r["n_raw_blobs"], whole=r["whole"], raw=sr, masked=sm,
                             p_raw=r["p_raw"].tolist(), p_masked=r["p_masked"].tolist()))
            print(f"{g:10s} {os.path.basename(f):14s} blobs={r['n_raw_blobs']:3d} cells={r['n_cells']:3d} "
                  f"whole={r['whole']:.3f} | raw: pos={sr['n_pos']:3d} ({sr['frac_pos']:.1%}) max={sr['max']:.3f} top3={sr['top3']:.3f} "
                  f"| masked: pos={sm['n_pos']:3d} ({sm['frac_pos']:.1%}) max={sm['max']:.3f} top3={sm['top3']:.3f}", flush=True)
            if g == "extra" and len(r["p_masked"]):
                tag = os.path.splitext(os.path.basename(f))[0]
                montage(raw, r["p_raw"], os.path.join(OUT, f"top_raw_{tag}.png"))
                montage(masked, r["p_masked"], os.path.join(OUT, f"top_masked_{tag}.png"))
        results[g] = rows
    with open(os.path.join(OUT, "tiling_eval.json"), "w") as fh:
        json.dump(results, fh)
