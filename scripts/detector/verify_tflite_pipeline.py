"""
Phase 13: end-to-end check of the DEPLOYED form of the pipeline in Python: letterbox 640 -> TFLite detector (float16 and float32)
-> decode (detector score >= 0.3, 3x3 peak pick, NO de-dup, exactly like the Phase 11c numbers) -> crops (box x1.15, native
resolution) -> the BUNDLED malaria_model.tflite -> field score = max cell score -> "any cell >= t" (t = 0.985).
Compared per field with the Keras reference scores from Phase 11c (artifacts/classifier_v2/cells_baseline.npz).
Fields: the 133 infected + 30 negative BBBC041 validation fields (never the reserved photos).
Timing printed here is DESKTOP CPU, one crop at a time; it is NOT a phone measurement.
Usage: python scripts/detector/verify_tflite_pipeline.py
"""
import os
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join("scripts", "bbbc041"))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join("scripts", "classifier_v2"))
import build_crops
from export_tflite import decode_prob, letterbox
from common import INFECTED, image_path, read_bgr, splits
from fp_bbbc_negatives import negatives

T_FIELD = 0.985
DET_THR = 0.3
ASSET = os.path.join("android", "app", "src", "main", "assets", "malaria_model.tflite")


class Tfl:
    def __init__(self, path=None, blob=None):
        self.it = tf.lite.Interpreter(model_path=path, model_content=blob) if blob is None else tf.lite.Interpreter(model_content=blob)
        self.it.allocate_tensors()
        self.i, self.o = self.it.get_input_details()[0], self.it.get_output_details()[0]

    def __call__(self, x):
        self.it.set_tensor(self.i["index"], x)
        self.it.invoke()
        return self.it.get_tensor(self.o["index"])


def auc(mx, neg):
    p, n = mx[~neg], mx[neg]
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


if __name__ == "__main__":
    sp = splits()["val"]
    neg_ids = {r["image"]["pathname"] for r in negatives(sp)}
    pos_ids = {r["image"]["pathname"] for r in sp if any(o["category"] in INFECTED for o in r["objects"])}
    recs = [r for r in sp if r["image"]["pathname"] in neg_ids | pos_ids]    # same order as the Phase 11c cache
    ref = np.load(os.path.join("artifacts", "classifier_v2", "maxscores_baseline.npz"))
    ref_mx, is_neg = ref["mx"], ref["is_neg"]
    assert len(recs) == len(ref_mx) and all((r["image"]["pathname"] in neg_ids) == n for r, n in zip(recs, is_neg))
    cls = Tfl(path=ASSET)
    build_crops.PAD = 1.15
    for kind in ("float16", "float32"):
        det = Tfl(path=os.path.join("model_output", "detector", f"cell_detector_{kind}.tflite"))
        mx, ncell, t_det, t_cls = [], [], [], []
        for r in recs:
            img = read_bgr(image_path(r))
            t0 = time.perf_counter()
            x, s = letterbox(img)
            boxes, sc = decode_prob(det(x[None])[0], DET_THR)
            t1 = time.perf_counter()
            boxes = boxes / s
            best = 0.0
            for b in boxes:
                c = build_crops.crop(img, {"minimum": {"r": b[1], "c": b[0]}, "maximum": {"r": b[3], "c": b[2]}})
                p = float(cls(((c.astype(np.float32) - 127.5) / 127.5)[None]).reshape(-1)[0])
                best = max(best, p)
            t2 = time.perf_counter()
            mx.append(best); ncell.append(len(boxes)); t_det.append(t1 - t0); t_cls.append(t2 - t1)
        mx = np.array(mx)
        d = np.abs(mx - ref_mx)
        flips = int(((mx >= T_FIELD) != (ref_mx >= T_FIELD)).sum())
        print(f"\n===== deployed pipeline, detector {kind} + bundled float32 classifier =====")
        print(f"  field max-score vs Keras reference (Phase 11c): mean|diff| {d.mean():.4f}, max|diff| {d.max():.3f}, decisions flipped at t={T_FIELD}: {flips}/{len(mx)}")
        print(f"  image AUC {auc(mx, is_neg):.3f} (reference {auc(ref_mx, is_neg):.3f})")
        for name, m in (("this pipeline", mx), ("Keras reference", ref_mx)):
            print(f"  {name:16s} @t={T_FIELD}: infected fields caught {(m[~is_neg] >= T_FIELD).sum()}/{int((~is_neg).sum())} = {(m[~is_neg] >= T_FIELD).mean():.1%}, "
                  f"negative fields flagged {(m[is_neg] >= T_FIELD).sum()}/{int(is_neg.sum())} = {(m[is_neg] >= T_FIELD).mean():.1%}")
        print(f"  cells per field: mean {np.mean(ncell):.1f}, max {max(ncell)}")
        print(f"  DESKTOP CPU timing (not a phone): detector {np.mean(t_det) * 1000:.0f} ms/field, classifier one crop at a time {np.mean(t_cls) * 1000:.0f} ms/field "
              f"({np.mean(t_cls) / max(np.mean(ncell), 1) * 1000:.1f} ms/crop)")
        np.savez(os.path.join("artifacts", "detector", f"tflite_pipeline_scores_{kind}.npz"), mx=mx, ncell=np.array(ncell), is_neg=is_neg)
