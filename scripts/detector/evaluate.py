"""
Detector evaluation against BBBC041 ground truth, recall split by infected / uninfected cells.
Usage: python scripts/detector/evaluate.py
Matching: a GT cell is "found" if a detection has IoU >= 0.5 with it (strict), and also reported with the Phase 9b
criterion (detection centre lies inside the GT box) so the numbers compare directly with the heuristic detector.
Score threshold is chosen on VALIDATION images only (highest threshold with val infected recall >= 90%,
else 0.3), then applied unchanged to the official test set. Nothing is tuned on test.
"""
import json
import os
import random
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
import train as T  # registers loss_fn for load_model
from common import OUT, decode, gt_boxes, image_path, iou_matrix, prep_eval, read_bgr, splits

INF_KEYS = None


def load():
    return tf.keras.models.load_model(os.path.join(OUT, "detector_best.keras"), custom_objects={"loss_fn": T.loss_fn}, compile=False)


def detect(model, rec):
    img = read_bgr(image_path(rec))
    x, s = prep_eval(img)
    out = model.predict(x, verbose=0)[0]
    return decode(out, score_thr=0.05), s   # low floor; thresholds applied later


def evaluate(model, recs, thr, only_infected_images=False):
    tot = {"inf": [0, 0, 0, 0], "rbc": [0, 0, 0, 0]}   # found@IoU.5, total, found@center, (unused)
    fp = n_det = 0
    per_img = []
    for rec in recs:
        (boxes, sc), s = detect(model, rec)
        keep = sc >= thr
        boxes, sc = boxes[keep] / s, sc[keep]
        gb, inf, cell = gt_boxes(rec)
        gb, inf = gb[cell], inf[cell]
        iou = iou_matrix(gb, boxes)
        cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
        for i, (b, is_inf) in enumerate(zip(gb, inf)):
            k = "inf" if is_inf else "rbc"
            tot[k][1] += 1
            tot[k][0] += bool(len(boxes) and iou[i].max() >= 0.5)
            tot[k][2] += bool(len(boxes) and ((cx >= b[0]) & (cx <= b[2]) & (cy >= b[1]) & (cy <= b[3])).any())
        n_det += len(boxes)
        fp += int((iou.max(0) < 0.5).sum()) if len(gb) and len(boxes) else len(boxes)
        per_img.append((rec, boxes, sc))
    return tot, fp, n_det, per_img


def fmt(tot, fp, n_det):
    r = lambda k, j: f"{tot[k][j]}/{tot[k][1]} = {tot[k][j] / max(tot[k][1], 1):.1%}"
    allf = tot["inf"][0] + tot["rbc"][0]; alln = tot["inf"][1] + tot["rbc"][1]
    allc = tot["inf"][2] + tot["rbc"][2]
    return (f"  recall ALL cells   IoU>=0.5: {allf}/{alln} = {allf / alln:.1%} | centre-in-box: {allc}/{alln} = {allc / alln:.1%}\n"
            f"  recall INFECTED    IoU>=0.5: {r('inf', 0)} | centre-in-box: {r('inf', 2)}\n"
            f"  recall uninfected  IoU>=0.5: {r('rbc', 0)} | centre-in-box: {r('rbc', 2)}\n"
            f"  detections {n_det}, unmatched (IoU<0.5 with any labelled cell) {fp} = {fp / max(n_det, 1):.1%}"
            " (unlabelled leukocytes / partial cells count here)")


if __name__ == "__main__":
    model = load()
    sp = splits()
    print("== threshold selection on validation (181 images) ==", flush=True)
    chosen = 0.3
    cache = {}
    for thr in (0.5, 0.4, 0.3, 0.2, 0.1):
        tot, fp, nd, _ = evaluate(model, sp["val"], thr)
        rec_inf = tot["inf"][0] / tot["inf"][1]
        print(f"thr {thr}: infected recall {rec_inf:.1%}, all {(tot['inf'][0] + tot['rbc'][0]) / (tot['inf'][1] + tot['rbc'][1]):.1%}, unmatched {fp / nd:.1%}", flush=True)
        cache[thr] = rec_inf
    ok = [t for t, r in cache.items() if r >= 0.90]
    chosen = max(ok) if ok else 0.3
    print(f"chosen threshold (highest with val infected recall >= 90%): {chosen}\n", flush=True)
    tot, fp, nd, _ = evaluate(model, sp["val"], chosen)
    print(f"== VALIDATION (thr {chosen}) ==\n{fmt(tot, fp, nd)}\n", flush=True)
    tot, fp, nd, _ = evaluate(model, sp["test"], chosen)
    print(f"== OFFICIAL TEST, all 120 images (different microscope setup; thr {chosen}) ==\n{fmt(tot, fp, nd)}\n", flush=True)
    # Same 30 test images as the heuristic-detector run in Phase 9b (seed 3, images containing infected cells)
    inf_names = {"trophozoite", "schizont", "gametocyte", "ring"}
    recs = [r for r in sp["test"] if any(o["category"] in inf_names for o in r["objects"])]
    recs = random.Random(3).sample(recs, 30)
    tot, fp, nd, _ = evaluate(model, recs, chosen)
    print(f"== SAME 30 TEST IMAGES AS PHASE 9b HEURISTIC RUN (thr {chosen}) ==\n{fmt(tot, fp, nd)}", flush=True)
    print("   heuristic detector on these images: infected 25/81 = 30.9%, uninfected 755/1371 = 55.1% (centre-in-box)")
