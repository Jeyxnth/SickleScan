"""
Phase 14c: how much does the phone's resize (validated emulation) vs TensorFlow's resize change the bundled models' outputs, over large image sets?
For each image: tensor A = original Python path (tf.io.decode_image + tf.image.resize bilinear + normalise), tensor B = same decoded pixels resized with the
Android emulation (1/16-pixel weights, truncation). Both go through the same desktop TFLite interpreter (identical to the phone runtime, shown on device).
Caveat: JPEG decoder differences (a few gray levels, Android vs TF libjpeg) are NOT included, only the resize difference.
Sets: sickle-cell images (all 569, sickle model, with the app's Positive/Borderline/Negative status rule), guardrail v3 on the Phase 14 evaluation sets.
Usage: python scripts/device/preprocessing_impact.py
"""
import csv, glob, os, sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join("scripts", "guardrail"))
sys.path.insert(0, os.path.join("scripts", "device"))
import data_pipeline as dp
from validate_android_resize_emulation import android_resize

ASSETS = os.path.join("android", "app", "src", "main", "assets")


def interp(f):
    it = tf.lite.Interpreter(model_path=os.path.join(ASSETS, f)); it.allocate_tensors(); return it


def prob(it, x):
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(i["index"], x[None].astype(np.float32)); it.invoke()
    return float(it.get_tensor(o["index"]).reshape(-1)[0])


def both(path_or_arr):
    if isinstance(path_or_arr, str):
        raw = tf.io.decode_image(tf.io.read_file(path_or_arr), channels=3, expand_animations=False).numpy()
    else:
        raw = path_or_arr
    a = dp.normalize(tf.image.resize(raw.astype(np.float32), [224, 224]).numpy())
    b = dp.normalize(android_resize(raw))
    return a, b


def status(p, ceiling=65.0):
    conf = (p if p >= .5 else 1 - p) * 100
    return "borderline" if conf < ceiling else ("positive" if p >= .5 else "negative")


def report(name, pa, pb, want_accept=None):
    pa, pb = np.array(pa), np.array(pb); d = np.abs(pa - pb)
    line = f"{name:58s} n={len(pa):5d}  |dP| mean {d.mean():.4f} max {d.max():.4f}  >0.03: {int((d > 0.03).sum()):4d}  decision flips at 0.5: {int(((pa >= .5) != (pb >= .5)).sum())}"
    print(line)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # ---- sickle model on the sickle images (the app's own status rule)
    sk = sorted(glob.glob("data/raw/Positive/Unlabelled/*.jpg")) + sorted(glob.glob("data/raw/Negative/Clear/*.jpg"))
    lab = [1] * len(glob.glob("data/raw/Positive/Unlabelled/*.jpg")) + [0] * len(glob.glob("data/raw/Negative/Clear/*.jpg"))
    m = interp("sicklescan_model.tflite")
    pa, pb = [], []
    for f in sk:
        a, b = both(f); pa.append(prob(m, a)); pb.append(prob(m, b))
    report("sickle model, all sickle images (TF resize vs phone resize)", pa, pb)
    sa, sb = [status(p) for p in pa], [status(p) for p in pb]
    print(f"   app status (Positive/Borderline/Negative, 65% band) changes: {sum(x != y for x, y in zip(sa, sb))}/{len(sa)}; "
          f"agreement with label at 0.5: TF {np.mean([(p >= .5) == l for p, l in zip(pa, lab)]):.4f} vs phone {np.mean([(p >= .5) == l for p, l in zip(pb, lab)]):.4f} "
          "(these images were also used to train the model, so this is parity, not held-out accuracy)")
    # ---- guardrail v3 on the Phase 14 evaluation sets
    g = interp("guardrail_model.tflite")
    rows = list(csv.DictReader(open("data/guardrail/manifest.csv", newline="", encoding="utf-8")))
    ev = list(csv.DictReader(open("data/guardrail/bbbc_eval.csv", newline="", encoding="utf-8")))
    sets = {
        "guardrail v3: BBBC held-out validation fields (163) [accept]": [r["path"] for r in ev if r["source"] == "bbbc041_val163"],
        "guardrail v3: BBBC official test images (120) [accept]": [r["path"] for r in ev if r["source"] == "bbbc041_test120"],
        "guardrail v3: sickle-cell test images (86) [accept]": [r["path"] for r in rows if r["split"] == "test" and r["source"] == "sickle_cell"],
        "guardrail v3: NIH single-cell test images (387) [accept]": [r["path"] for r in rows if r["split"] == "test" and r["source"] == "malaria"],
        "guardrail v3: HELD-OUT real non-smear (945) [reject]": [r["path"] for r in rows if r["label"] == "not_smear" and r["split"] in ("val", "test") and not r["source"].startswith("synthetic")],
        "guardrail v3: FRESH non-smear (986) [reject]": sorted(glob.glob("data/guardrail/fresh_negatives/*/*.jpg")),
    }
    for name, paths in sets.items():
        pa, pb = [], []
        for p in paths:
            a, b = both(p); pa.append(prob(g, a)); pb.append(prob(g, b))
        report(name, pa, pb)
    z = np.load(os.path.join("artifacts", "bbbc041", "crops_val.npz"), allow_pickle=True)["X"]
    idx = rng.choice(len(z), 300, replace=False)
    pa, pb = [], []
    for i in idx:   # 224x224 crops: the app's resize is 224 -> 224
        a, b = both(z[i]); pa.append(prob(g, a)); pb.append(prob(g, b))
    report("guardrail v3: BBBC single-cell crops, 224 -> 224 (300 of 1,823)", pa, pb)
