"""
Crop each detected cell, resize to 224x224, and score it with the existing malaria Keras model.

Two crop styles are evaluated because the malaria training crops (NIH dataset) are single cells
segmented onto a BLACK background, while a raw crop from a field photo also contains neighbouring
cells and light background:
  raw    : square crop around the cell (cell bbox + padding)
  masked : same crop, but every pixel outside the detected cell mask set to black (matches training look)
Preprocessing after resize is identical to the malaria pipeline: bilinear 224x224, (x-127.5)/127.5, RGB.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from detect_cells import detect_cells, load_bgr

PAD = 0.12  # fraction of the cell bbox added on each side -> cell fills ~80% of the frame


def make_crops(det, img_bgr):
    """Returns (cells, raw_crops, masked_crops) for kept cells; crops are 224x224 RGB uint8."""
    inv = 1.0 / det["scale"]
    H, W = img_bgr.shape[:2]
    cells, raw, masked = [], [], []
    for c in det["cells"]:
        if not c["kept"]:
            continue
        x, y, bw, bh = c["bbox"]
        side = max(bw, bh) * (1 + 2 * PAD)
        cxs, cys = x + bw / 2.0, y + bh / 2.0
        # crop box in full-res coordinates
        s = max(8, int(round(side * inv)))
        x0, y0 = int(round(cxs * inv - s / 2)), int(round(cys * inv - s / 2))
        canvas = np.zeros((s, s, 3), np.uint8)
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + s, W), min(y0 + s, H)
        if xb <= xa or yb <= ya:
            continue
        canvas[ya - y0:yb - y0, xa - x0:xb - x0] = img_bgr[ya:yb, xa:xb]
        # matching crop of the cell mask, taken at working scale then resized to the crop size
        ss = max(4, int(round(side)))
        sx0, sy0 = int(round(cxs - ss / 2)), int(round(cys - ss / 2))
        mcanvas = np.zeros((ss, ss), np.uint8)
        h_, w_ = c["mask"].shape
        xa2, ya2, xb2, yb2 = max(sx0, 0), max(sy0, 0), min(sx0 + ss, w_), min(sy0 + ss, h_)
        mcanvas[ya2 - sy0:yb2 - sy0, xa2 - sx0:xb2 - sx0] = c["mask"][ya2:yb2, xa2:xb2] * 255
        mfull = cv2.resize(mcanvas, (s, s), interpolation=cv2.INTER_NEAREST)
        mfull = cv2.dilate(mfull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        rawc = cv2.resize(canvas, (224, 224), interpolation=cv2.INTER_LINEAR)
        mc = canvas.copy()
        mc[mfull == 0] = 0
        maskc = cv2.resize(mc, (224, 224), interpolation=cv2.INTER_LINEAR)
        cells.append(c)
        raw.append(cv2.cvtColor(rawc, cv2.COLOR_BGR2RGB))
        masked.append(cv2.cvtColor(maskc, cv2.COLOR_BGR2RGB))
    return cells, raw, masked


def to_model_input(crops):
    return (np.stack(crops).astype(np.float32) - 127.5) / 127.5


def whole_image_input(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    return (r.astype(np.float32)[None] - 127.5) / 127.5


def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(os.path.join("artifacts", "malaria", "malaria_keras.keras"))


def classify_field(model, path):
    img = load_bgr(path)
    det = detect_cells(img)
    cells, raw, masked = make_crops(det, img)
    out = dict(path=path, n_raw_blobs=len(det["cells"]), n_cells=len(cells),
               whole=float(model.predict(whole_image_input(img), verbose=0)[0, 0]))
    if cells:
        out["p_raw"] = model.predict(to_model_input(raw), batch_size=64, verbose=0).reshape(-1)
        out["p_masked"] = model.predict(to_model_input(masked), batch_size=64, verbose=0).reshape(-1)
    else:
        out["p_raw"] = out["p_masked"] = np.zeros(0)
    out["_crops"] = (raw, masked, cells)
    return out


def summarize(p, thr=0.5):
    if len(p) == 0:
        return dict(n=0, n_pos=0, frac_pos=0.0, max=0.0, top3=0.0, n_ge_0p9=0)
    s = np.sort(p)[::-1]
    return dict(n=int(len(p)), n_pos=int((p >= thr).sum()), frac_pos=float((p >= thr).mean()),
                max=float(s[0]), top3=float(s[:3].mean()), n_ge_0p9=int((p >= 0.9).sum()))
