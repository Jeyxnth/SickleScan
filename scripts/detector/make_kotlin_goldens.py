"""
Golden values for the Kotlin unit tests (ImageOpsTest): the SAME OpenCV code used in the Phase 11 validation
(export_tflite.letterbox, build_crops.crop) run on a deterministic synthetic image that Kotlin regenerates exactly:
    value(x, y, c) = (x*7 + y*13 + ((x*y) % 37)*5 + c*50) % 256      (c = 0,1,2 -> R, G, B)
Writes android/app/src/test/resources/goldens/{letterbox,crops}.txt (small, synthetic -- no dataset or user images).
Usage: python scripts/detector/make_kotlin_goldens.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join("scripts", "bbbc041"))
sys.path.insert(0, os.path.dirname(__file__))
import build_crops
import export_tflite

OUT = os.path.join("android", "app", "src", "test", "resources", "goldens")


def synth(w, h):
    y, x = np.mgrid[0:h, 0:w]
    rgb = np.stack([(x * 7 + y * 13 + ((x * y) % 37) * 5 + c * 50) % 256 for c in range(3)], -1).astype(np.uint8)
    return rgb


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    lines = []
    for (w, h, size) in ((53, 41, 32), (41, 53, 32), (53, 41, 128), (40, 40, 40)):
        rgb = synth(w, h)
        export_tflite.SIZE = size
        f, s = export_tflite.letterbox(rgb[..., ::-1].copy())     # letterbox expects BGR
        canvas = np.rint(f * 127.5 + 127.5).astype(np.int32)
        lines.append(f"letterbox {w} {h} {size} scale={s:.9f}")
        lines.append(" ".join(map(str, canvas.reshape(-1))))
    open(os.path.join(OUT, "letterbox.txt"), "w").write("\n".join(lines) + "\n")

    W, H = 300, 260
    rgb = synth(W, H)
    bgr = rgb[..., ::-1].copy()
    build_crops.PAD = 1.15
    cases = {
        "interior_small": (100.0, 90.0, 140.0, 132.0),      # side ~48 -> enlarge (INTER_LINEAR)
        "interior_large": (40.0, 30.0, 250.0, 240.0),       # side ~242 -> shrink (INTER_AREA)
        "left_edge": (-10.0, 100.0, 40.0, 150.0),           # window sticks out of the image -> replicated border
        "bottom_right": (270.0, 235.0, 310.0, 275.0),
        "tiny": (150.0, 120.0, 155.0, 127.0),               # side < 16 -> clamped to 16
        "half_pixel": (100.4, 90.6, 140.9, 131.2),          # rounding of a fractional centre / side
    }
    out = [f"image {W} {H}"]
    for name, (x0, y0, x1, y1) in cases.items():
        c = build_crops.crop(bgr, {"minimum": {"r": np.float32(y0), "c": np.float32(x0)}, "maximum": {"r": np.float32(y1), "c": np.float32(x1)}})
        sub = c[::7, ::7].astype(np.int32)                    # 32 x 32 x 3 subsample
        out.append(f"{name} {x0} {y0} {x1} {y1} sum={int(c.astype(np.int64).sum())}")
        out.append(" ".join(map(str, sub.reshape(-1))))
    open(os.path.join(OUT, "crops.txt"), "w").write("\n".join(out) + "\n")
    print("wrote goldens", [os.path.getsize(os.path.join(OUT, f)) for f in ("letterbox.txt", "crops.txt")])
