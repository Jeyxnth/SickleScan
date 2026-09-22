"""Phase 14d: exact effect of the new (correct, phone-verified) preprocessing on the reported 86-image sickle test-set metrics
(accuracy / sensitivity / specificity in README.md), computed from real on-device probabilities (demo_images/device_checks/parity/results.csv),
compared against the TensorFlow reference (what the README's numbers were computed with).
Usage: python scripts/device/test_split_impact.py"""
import csv
import os

import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, os.path.join("scripts", "guardrail"))
import data_pipeline as dp

P = os.path.join("demo_images", "device_checks", "parity")
res = {(r["path"], r["model"]): float(r["P"]) for r in csv.DictReader(open(os.path.join(P, "results.csv")))}
it = tf.lite.Interpreter(model_path=os.path.join("android", "app", "src", "main", "assets", "sicklescan_model.tflite")); it.allocate_tensors()
i_, o_ = it.get_input_details()[0], it.get_output_details()[0]


def tf_prob(path):
    it.set_tensor(i_["index"], dp.load_and_preprocess_single(path)[None].astype(np.float32)); it.invoke()
    return float(it.get_tensor(o_["index"]).reshape(-1)[0])


rows = list(csv.DictReader(open(os.path.join("data", "splits", "test.csv"), newline="", encoding="utf-8")))
print(f"test split: {len(rows)} images ({sum(r['label'] == 'positive' for r in rows)} positive, {sum(r['label'] == 'negative' for r in rows)} negative)")


def metrics(preds, labels):
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels)); tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels)); fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
    n = len(labels)
    return dict(acc=(tp + tn) / n, sens=tp / (tp + fn) if tp + fn else float("nan"), spec=tn / (tn + fp) if tn + fp else float("nan"), tp=tp, tn=tn, fp=fp, fn=fn)


labels, p_tf, p_phone = [], [], []
for r in rows:
    path = r["path"].replace("\\", "/")
    lbl = "positive" if "Positive" in path else "negative"
    rel = f"sickle/{lbl}__{os.path.basename(path)}"
    pd_ = res.get((rel, "sickle"))
    assert pd_ is not None, f"missing phone result for {rel}"
    labels.append(1 if r["label"] == "positive" else 0)
    p_tf.append(tf_prob(path))
    p_phone.append(pd_)

for name, p in (("TensorFlow preprocessing (README's current numbers)", p_tf), ("Phone / new resize (as the app now behaves)", p_phone)):
    preds = [1 if x >= .5 else 0 for x in p]
    m = metrics(preds, labels)
    print(f"{name}: accuracy {m['acc']:.4%}  sensitivity {m['sens']:.4%}  specificity {m['spec']:.4%}  (TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']})")

diffs = [(rows[k]["path"], labels[k], p_tf[k], p_phone[k]) for k in range(len(rows)) if (p_tf[k] >= .5) != (p_phone[k] >= .5)]
print(f"\nimages whose test-set decision changes ({len(diffs)}):")
for path, lbl, pt, pp in diffs:
    print(f"  {path} (true label {'positive' if lbl else 'negative'}): TensorFlow {pt:.4f} -> phone {pp:.4f}")
