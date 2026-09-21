"""Run the cell detector on a set of field images; print stats and write overlays to artifacts/tiling/."""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from detect_cells import detect_cells, hough_count, load_bgr, overlay

OUT = os.path.join("artifacts", "tiling")
os.makedirs(OUT, exist_ok=True)


def summarize(name, det, hough_n):
    cells = det["cells"]
    kept = [c for c in cells if c["kept"]]
    big = [c for c in cells if c["rel_area"] > 2.0]
    small = [c for c in cells if c["rel_area"] < 0.45]
    edge = [c for c in cells if c["touches_edge"]]
    areas = np.array([c["area"] for c in kept]) if kept else np.array([0.0])
    print(f"{name:26s} raw={len(cells):3d} kept={len(kept):3d} big(>2x)={len(big):2d} tiny(<.45x)={len(small):2d} "
          f"edge={len(edge):2d} r_est={det['r_est']:.1f}px medArea={det['median_area']:.0f} "
          f"kept-area CV={areas.std() / max(areas.mean(), 1):.2f} hough={hough_n}")


if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "otsu"
    sets = {
        "sickle_pos": sorted(glob.glob("data/raw/Positive/Unlabelled/*.jpg"))[:6],
        "clear": sorted(glob.glob("data/raw/Negative/Clear/*.jpg"))[:6],
    }
    for group, files in sets.items():
        print(f"\n== {group} ({method}) ==")
        for f in files:
            name = f"{group}_{os.path.basename(f)}"
            det = detect_cells(load_bgr(f), method)
            hn = hough_count(load_bgr(f), det["r_est"])
            summarize(name, det, hn)
            overlay(det, os.path.join(OUT, f"ov_{method}_{name}.png"))
