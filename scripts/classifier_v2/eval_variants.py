"""
Compare crop classifiers on the SAME held-out BBBC041 validation fields with the SAME detector boxes.
Negative fields = 30 val fields with no infected / difficult box; positive fields = 133 val fields with infected cells.
Detector boxes (thr 0.2) are computed once and cached. Reports, per variant:
  - image-level ROC AUC of max cell score, positives vs negatives (threshold-free; primary metric)
  - at the app's fixed 0.5 boundary: per-cell flag rate on negative fields, negative fields flagged, positive fields caught
    (rule: any cell >= 0.5), with and without box de-duplication (NMS IoU 0.5 on detector boxes)
  - the pre-agreed target check: best negative-flag rate among thresholds with positive-field sensitivity >= 90%
    (NOTE: threshold chosen on these same fields -> optimistic; shown only as a diagnostic, not as a validated number)
Usage: python scripts/classifier_v2/eval_variants.py NAME=PATH:PAD [NAME=PATH:PAD ...]   (PATH 'baseline' = current model)
"""
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join("scripts", "bbbc041"))
sys.path.insert(0, os.path.join("scripts", "detector"))
import build_crops
import pipeline as P
from common import INFECTED, decode, image_path, iou_matrix, prep_eval, read_bgr, splits
from fp_bbbc_negatives import negatives

CACHE = os.path.join("artifacts", "classifier_v2", "val_detections.npz")
BASE = os.path.join("artifacts", "bbbc041", "model", "bbbc_keras.keras")


def detections(det, recs):
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return list(z["boxes"]), list(z["scores"])
    B, S = [], []
    for r in recs:
        x, s = prep_eval(read_bgr(image_path(r)))
        b, sc = decode(det.predict(x, verbose=0)[0], score_thr=0.2)
        B.append(b / s); S.append(sc)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, boxes=np.array(B, dtype=object), scores=np.array(S, dtype=object))
    return B, S


def nms(b, sc, thr=0.5):
    order = np.argsort(-sc); keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        order = rest[iou_matrix(b[i:i + 1], b[rest])[0] < thr]
    return np.array(keep, int)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def crops_for(img, boxes, pad):
    build_crops.PAD = pad
    return np.stack([P.crop(img, {"minimum": {"r": b[1], "c": b[0]}, "maximum": {"r": b[3], "c": b[2]}}) for b in boxes]) if len(boxes) else np.zeros((0, 224, 224, 3), np.uint8)


if __name__ == "__main__":
    sp = splits()["val"]
    neg_ids = {r["image"]["pathname"] for r in negatives(sp)}
    pos_ids = {r["image"]["pathname"] for r in sp if any(o["category"] in INFECTED for o in r["objects"])}
    recs = [r for r in sp if r["image"]["pathname"] in neg_ids | pos_ids]
    det = tf.keras.models.load_model(os.path.join("artifacts", "detector", "detector_best.keras"),
                                     custom_objects={"loss_fn": P.T.loss_fn}, compile=False) if not os.path.exists(CACHE) else None
    Bx, Sx = detections(det, recs)
    is_neg = np.array([r["image"]["pathname"] in neg_ids for r in recs])
    for spec_ in sys.argv[1:]:
        name, rest = spec_.split("=", 1)
        path, pad = rest.rsplit(":", 1)
        pad = float(pad)
        model = tf.keras.models.load_model(BASE if path == "baseline" else path)
        scores = []
        for r, b, sc in zip(recs, Bx, Sx):
            img = read_bgr(image_path(r))
            c = crops_for(img, b, pad)
            p = model.predict((c.astype(np.float32) - 127.5) / 127.5, batch_size=64, verbose=0).reshape(-1) if len(c) else np.zeros(0)
            scores.append(p)
        np.savez(os.path.join("artifacts", "classifier_v2", f"cells_{name}.npz"), p=np.array(scores, dtype=object),
                 sc=np.array(Sx, dtype=object), is_neg=is_neg)
        print(f"\n##### {name} (crop padding {pad}) -- {int((~is_neg).sum())} infected fields, {int(is_neg.sum())} negative fields")
        for dedup in (False, True):
            for thr in (0.3,):
                ps = []
                for b, sc, p in zip(Bx, Sx, scores):
                    k = sc >= thr
                    b_, sc_, p_ = b[k], sc[k], p[k]
                    if dedup and len(b_):
                        kk = nms(b_, sc_); p_ = p_[kk]
                    ps.append(p_)
                mx = np.array([q.max() if len(q) else 0 for q in ps])
                if not dedup:
                    np.savez(os.path.join("artifacts", "classifier_v2", f"maxscores_{name}.npz"), mx=mx, is_neg=is_neg)
                a = auc(mx[~is_neg], mx[is_neg])
                cells_neg = np.concatenate([q for q, n in zip(ps, is_neg) if n])
                fl = int((cells_neg >= .5).sum())
                caught = int((mx[~is_neg] >= .5).sum()); nflag = int((mx[is_neg] >= .5).sum())
                best = None
                for t in np.unique(np.round(np.concatenate([mx, [0.5, 0.9, 0.99]]), 4)):
                    sens = (mx[~is_neg] >= t).mean(); fpr = (mx[is_neg] >= t).mean()
                    if sens >= 0.90 and (best is None or fpr < best[1]):
                        best = (t, fpr, sens)
                print(f" detector thr {thr}, box de-dup {'ON ' if dedup else 'off'}: image AUC {a:.3f} | @0.5: per-cell flag on neg fields {fl}/{len(cells_neg)} = "
                      f"{fl / len(cells_neg):.2%}, neg fields flagged {nflag}/{int(is_neg.sum())} ({nflag / is_neg.sum():.0%}), pos fields caught {caught}/{int((~is_neg).sum())} "
                      f"({caught / (~is_neg).sum():.0%}) | diag (in-sample thr): sens>=90% -> min neg-flag {best[1]:.0%} at t={best[0]:.3f}" if best else
                      f" detector thr {thr}, dedup {dedup}: image AUC {a:.3f}; no threshold reaches 90% sensitivity")
