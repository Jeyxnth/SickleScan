"""
Phase 13 Task 2: does the wide-field detector still work when the whole photo is ONE zoomed-in cell?
Each 224x224 validation crop (ground-truth cell box x1.15, the kind of image the single-cell path is built for) is treated as a
whole photo: letterbox to 640 (so the cell fills ~87% of the frame, i.e. ~550 px vs 22-65 px cells the detector was trained on),
run the TFLite detector (float32, thr 0.3), then compare with the existing direct single-cell classification of the same image.
Uses validation crops only (never the reserved photos). Usage: python scripts/detector/single_cell_test.py
"""
import glob
import os
import random
import sys

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join("scripts", "bbbc041"))
sys.path.insert(0, os.path.dirname(__file__))
import build_crops
from export_tflite import decode_prob, letterbox
from verify_tflite_pipeline import Tfl, ASSET
from common import iou_matrix

if __name__ == "__main__":
    det = Tfl(path=os.path.join("model_output", "detector", "cell_detector_float32.tflite"))
    cls = Tfl(path=ASSET)
    z = np.load(os.path.join("artifacts", "bbbc041", "crops_val.npz"), allow_pickle=True)
    X, y, cats = z["X"], z["y"], z["cats"]
    rng = random.Random(0)
    inf_idx = [i for i in range(len(y)) if y[i] == 1]
    unf_idx = [i for i in range(len(y)) if y[i] == 0 and cats[i] == "red blood cell"]
    sel = rng.sample(inf_idx, 100) + rng.sample(unf_idx, 100)
    # the cell occupies the central 1/1.15 of the crop (box x1.15 padding)
    m = 0.5 * (1 - 1 / 1.15)
    gt = np.array([[m * 224, m * 224, (1 - m) * 224, (1 - m) * 224]], np.float32)
    n_det, found, single, direct, via_det = [], 0, 0, [], []
    for i in sel:
        rgb = X[i]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        x, s = letterbox(bgr)
        boxes, sc = decode_prob(det(x[None])[0], 0.3)
        boxes = boxes / s
        n_det.append(len(boxes))
        ok = len(boxes) and iou_matrix(boxes, gt)[:, 0].max() >= 0.5
        found += bool(ok)
        if ok and len(boxes) == 1:
            single += 1
        p_direct = float(cls(((rgb.astype(np.float32) - 127.5) / 127.5)[None]).reshape(-1)[0])
        # detect-then-classify on this photo: max over detected cells' crops
        best = 0.0
        for b in boxes:
            c = build_crops.crop(bgr, {"minimum": {"r": b[1], "c": b[0]}, "maximum": {"r": b[3], "c": b[2]}})
            best = max(best, float(cls(((c.astype(np.float32) - 127.5) / 127.5)[None]).reshape(-1)[0]))
        direct.append(p_direct); via_det.append(best if len(boxes) else 0.0)
    direct, via_det, n_det = np.array(direct), np.array(via_det), np.array(n_det)
    truth = np.array([y[i] for i in sel]) >= .5
    print(f"{len(sel)} single-cell validation images (100 infected, 100 uninfected RBC), letterboxed to 640")
    print(f"detections per image: mean {n_det.mean():.2f}, zero detections {int((n_det == 0).sum())}/{len(sel)}, exactly one {int((n_det == 1).sum())}, >=2 {int((n_det >= 2).sum())}")
    print(f"the ONE cell found (a detection with IoU >= 0.5 against the true cell box): {found}/{len(sel)} = {found / len(sel):.0%}; found as the only detection: {single}/{len(sel)}")
    print("classification of the same images:")
    for name, p in (("direct single-cell path (existing)", direct), ("detector -> crop -> classify (max over cells)", via_det)):
        tp = int(((p >= .5) & truth).sum()); fp = int(((p >= .5) & ~truth).sum())
        print(f"  {name:48s} infected called positive at 0.5: {tp}/100, uninfected called positive: {fp}/100")
    print(f"  images where the two paths disagree at 0.5: {int(((direct >= .5) != (via_det >= .5)).sum())}/{len(sel)}")
    # the 12 local demo crops (BBBC041 val crops saved earlier), if present
    demo = sorted(glob.glob(os.path.join("demo_images", "malaria", "*.png")) + glob.glob(os.path.join("demo_images", "malaria", "*.jpg")))
    if demo:
        cnt = []
        for f in demo:
            bgr = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
            x, s = letterbox(bgr)
            cnt.append(len(decode_prob(det(x[None])[0], 0.3)[0]))
        print(f"local demo crops ({len(demo)}): detections per image {cnt}")
