"""
Phase 13 integration check: does the bundled Phase 7 guardrail (smear vs not-smear, accept if P(smear) >= 0.5) accept
wide-field BBBC041 validation photos? The app runs it before any disease model, and a rejection puts the result behind
a "Continue anyway" override and excludes it from dashboard stats. Uses the app's own preprocessing (whole image resized to
224x224 bilinear, (x-127.5)/127.5). Validation fields only. Usage: python scripts/detector/guardrail_on_fields.py
"""
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join("scripts", "bbbc041"))
sys.path.insert(0, os.path.join("scripts", "detector"))
sys.path.insert(0, os.path.join("scripts", "classifier_v2"))
from common import INFECTED, image_path, read_bgr, splits
from fp_bbbc_negatives import negatives

if __name__ == "__main__":
    it = tf.lite.Interpreter(model_path=os.path.join("android", "app", "src", "main", "assets", "guardrail_model.tflite"))
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    sp = splits()["val"]
    neg_ids = {r["image"]["pathname"] for r in negatives(sp)}
    pos_ids = {r["image"]["pathname"] for r in sp if any(c["category"] in INFECTED for c in r["objects"])}
    scores = {"infected": [], "negative": []}
    for r in sp:
        pid = r["image"]["pathname"]
        if pid not in neg_ids | pos_ids:
            continue
        rgb = cv2.cvtColor(read_bgr(image_path(r)), cv2.COLOR_BGR2RGB)
        x = (cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) - 127.5) / 127.5
        it.set_tensor(i["index"], x[None]); it.invoke()
        scores["negative" if pid in neg_ids else "infected"].append(float(it.get_tensor(o["index"]).reshape(-1)[0]))
    for k, v in scores.items():
        v = np.array(v)
        print(f"{k:9s} fields: {len(v)}, guardrail accepts (P(smear) >= 0.5): {(v >= .5).sum()}/{len(v)} = {(v >= .5).mean():.1%}; "
              f"rejected {(v < .5).sum()}; min score {v.min():.3f}, median {np.median(v):.3f}")
