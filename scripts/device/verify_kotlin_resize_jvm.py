"""
Phase 14d: verify the app's NEW classifier preprocessing (ImageOps.classifierInput: TensorFlow-exact bilinear, float, + normalise) against
TensorFlow's own preprocessing, using the exact Kotlin code (run on the JVM via a unit-test harness) on the pixels TensorFlow decoded, so the resize
algorithm is isolated from decoder differences. For every image the Kotlin tensor and the TF tensor go through the real bundled models (desktop
TFLite = the same runtime as the phone; shown in Phase 14c). Sets: the 25 test assets (all three models), all 569 sickle images (sickle model, with
the app's Positive/Borderline/Negative rule), NIH single-cell test crops (malaria + guardrail), BBBC fields / test images / non-smear sets (guardrail),
BBBC single-cell crops (malaria + guardrail).
Usage: python scripts/device/verify_kotlin_resize_jvm.py
"""
import csv
import glob
import os
import shutil
import struct
import subprocess
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join("scripts", "guardrail"))
import data_pipeline as dp

ASSETS = os.path.join("android", "app", "src", "main", "assets")
TMP = "D:/AI/temp/claude/parity_jvm"
CHUNK = 60
MODELS = {"sickle": "sicklescan_model.tflite", "malaria": "malaria_model.tflite", "guardrail": "guardrail_model.tflite"}
_it = {}


def interp(name):
    if name not in _it:
        it = tf.lite.Interpreter(model_path=os.path.join(ASSETS, MODELS[name])); it.allocate_tensors(); _it[name] = it
    return _it[name]


def prob(name, x):
    it = interp(name); i, o = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(i["index"], x[None].astype(np.float32)); it.invoke()
    return float(it.get_tensor(o["index"]).reshape(-1)[0])


def decode(item):
    if isinstance(item, np.ndarray):
        return item
    return tf.io.decode_image(tf.io.read_file(item), channels=3, expand_animations=False).numpy()


def run_chunk(items):
    """items: list of (key, path_or_array). Returns {key: (tf_tensor, kotlin_tensor)}."""
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP + "/in"); os.makedirs(TMP + "/out")
    tf_t = {}
    for k, (key, it) in enumerate(items):
        rgb = decode(it)
        with open(f"{TMP}/in/{k:05d}.raw", "wb") as f:
            f.write(struct.pack("<ii", rgb.shape[1], rgb.shape[0])); f.write(rgb.astype(np.uint8).tobytes())
        tf_t[k] = dp.normalize(tf.image.resize(rgb.astype(np.float32), [224, 224]).numpy())
    env = dict(os.environ, PARITY_IN=TMP + "/in", PARITY_OUT=TMP + "/out")
    r = subprocess.run([os.path.abspath(os.path.join("android", "gradlew.bat")), "cleanTestDebugUnitTest", "testDebugUnitTest", "--tests", "*ClassifierPreprocessJvmHarnessTest", "--console=plain", "-q"],
                       cwd="android", env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:]); raise SystemExit("gradle harness failed")
    out = {}
    for k, (key, _) in enumerate(items):
        out[key] = (tf_t[k], np.fromfile(f"{TMP}/out/{k:05d}.f32", "<f4").reshape(224, 224, 3))
    return out


def status(p, ceiling=65.0):
    conf = (p if p >= .5 else 1 - p) * 100
    return "borderline" if conf < ceiling else ("positive" if p >= .5 else "negative")


results = {}   # (set, model) -> list of (key, p_tf, p_kotlin)


def evaluate(set_name, items, models):
    for s in range(0, len(items), CHUNK):
        out = run_chunk(items[s:s + CHUNK])
        for key, (a, b) in out.items():
            for m in models:
                results.setdefault((set_name, m), []).append((key, prob(m, a), prob(m, b)))
    print(f"  done {set_name}: {len(items)} images x {models}", flush=True)


if __name__ == "__main__":
    A = os.path.join("android", "app", "src", "androidTest", "assets")
    assets = [(f"{d}/{n}", os.path.join(A, d, n)) for d in ("sample_test_images", "malaria_test_images", "guardrail_test_images") for n in sorted(os.listdir(os.path.join(A, d)))]
    evaluate("25 test assets", assets, ["sickle", "malaria", "guardrail"])
    sk = sorted(glob.glob("data/raw/Positive/Unlabelled/*.jpg")) + sorted(glob.glob("data/raw/Negative/Clear/*.jpg"))
    evaluate("sickle images (569)", [(p, p) for p in sk], ["sickle"])
    rows = list(csv.DictReader(open("data/guardrail/manifest.csv", newline="", encoding="utf-8")))
    ev = list(csv.DictReader(open("data/guardrail/bbbc_eval.csv", newline="", encoding="utf-8")))
    nih = [r["path"] for r in rows if r["split"] == "test" and r["source"] == "malaria"]
    evaluate("NIH single-cell test crops (387)", [(p, p) for p in nih], ["malaria", "guardrail"])
    z = np.load("artifacts/bbbc041/crops_val.npz", allow_pickle=True)["X"]
    idx = np.random.default_rng(0).choice(len(z), 200, replace=False)
    evaluate("BBBC single-cell crops 224x224 (200)", [(f"crop{i}", z[i]) for i in idx], ["malaria", "guardrail"])
    for nm, paths in (("BBBC held-out fields (163)", [r["path"] for r in ev if r["source"] == "bbbc041_val163"]),
                      ("BBBC official test images (120)", [r["path"] for r in ev if r["source"] == "bbbc041_test120"]),
                      ("held-out non-smear (945)", [r["path"] for r in rows if r["label"] == "not_smear" and r["split"] in ("val", "test") and not r["source"].startswith("synthetic")]),
                      ("fresh non-smear (986)", sorted(glob.glob("data/guardrail/fresh_negatives/*/*.jpg")))):
        evaluate(nm, [(p, p) for p in paths], ["guardrail"])
    shutil.rmtree(TMP, ignore_errors=True)

    print(f"\n{'set / model':56s} {'n':>5s} | |dP| mean    max   >0.001 | decision flips@0.5 | app status changes")
    for (sn, m), v in results.items():
        pa, pb = np.array([x[1] for x in v]), np.array([x[2] for x in v]); d = np.abs(pa - pb)
        flips = int(((pa >= .5) != (pb >= .5)).sum())
        chg = sum(status(x) != status(y) for x, y in zip(pa, pb)) if m in ("sickle", "malaria") else "-"
        print(f"{sn + ' / ' + m:56s} {len(v):5d} | {d.mean():.6f} {d.max():.6f} {int((d > 1e-3).sum()):6d} | {flips:^18d} | {chg}")
    t = {k: (a, b) for k, a, b in results[("25 test assets", "sickle")]}["malaria_test_images/parasitized_1.png"]
    print(f"\nphase14c test image (parasitized_1.png) on the SICKLE model: TensorFlow preprocessing {t[0]:.4f} | new Kotlin preprocessing {t[1]:.4f} | (old Android resize on the phone: 0.8515)")
