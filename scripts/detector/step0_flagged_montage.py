"""Step 0: montage of every classifier>=0.5 detection on the 30 BBBC041 val malaria-negative fields (detector thr 0.2).
Each tile: 2x-context crop (so neighbours are visible), red box = the detected cell, caption = score + what it overlaps.
Usage: python scripts/detector/step0_flagged_montage.py"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join("scripts", "bbbc041"))
sys.path.insert(0, os.path.dirname(__file__))
import pipeline as P
from common import gt_boxes, image_path, iou_matrix, read_bgr, splits
from fp_bbbc_negatives import negatives

TILE = 200

if __name__ == "__main__":
    det, cls = P.load_models()
    tiles, rows = [], []
    for r in negatives(splits()["val"]):
        img = read_bgr(image_path(r))
        b, sc, p = P.run(det, cls, img, 0.2)
        gb, inf, cell = gt_boxes(r)
        cats = np.array([o["category"] for o in r["objects"]])
        iou = iou_matrix(b, gb)
        for i in np.where(p >= .5)[0]:
            j = iou[i].argmax() if iou.shape[1] else -1
            what = "no label" if (j < 0 or iou[i, j] < 0.3) else ("RBC" if cats[j] != "leukocyte" else "WBC")
            x0, y0, x1, y1 = b[i]
            side = int(max(x1 - x0, y1 - y0) * 2.2)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            pad = side
            big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            ox, oy = int(cx - side / 2) + pad, int(cy - side / 2) + pad
            t = big[oy:oy + side, ox:ox + side].copy()
            s = TILE / side
            t = cv2.resize(t, (TILE, TILE))
            bx = [int((x0 - (cx - side / 2)) * s), int((y0 - (cy - side / 2)) * s), int((x1 - (cx - side / 2)) * s), int((y1 - (cy - side / 2)) * s)]
            cv2.rectangle(t, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 1)
            cv2.putText(t, f"{p[i]:.2f} {what}", (4, 16), 0, 0.5, (0, 0, 0), 3)
            cv2.putText(t, f"{p[i]:.2f} {what}", (4, 16), 0, 0.5, (0, 255, 255), 1)
            tiles.append(t)
            rows.append((os.path.basename(image_path(r)), float(p[i]), what))
    print(len(tiles), "flagged cells")
    while len(tiles) % 7:
        tiles.append(np.full((TILE, TILE, 3), 255, np.uint8))
    m = np.vstack([np.hstack(tiles[i:i + 7]) for i in range(0, len(tiles), 7)])
    cv2.imencode(".png", m)[1].tofile(os.path.join("model_output", "detector", "step0_flagged_cells.png"))
    print(m.shape)
