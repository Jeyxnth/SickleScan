"""
Phase 14d device parity suite, two subcommands.
  build   -> writes demo_images/device_checks/parity/{jobs.csv, files...} (gitignored; push to files/parity on the phone)
  analyze -> compares the phone's results.csv (pulled next to jobs.csv) with Python's TensorFlow preprocessing through the same bundled models.
Sets: the 25 test assets (all three models), all 569 sickle images (sickle), NIH single-cell test crops (malaria, guardrail), 200 BBBC single-cell crops
(malaria, guardrail), 163 BBBC fields + 120 BBBC test images (guardrail), held-out (945) and fresh (986) non-smear (guardrail).
Usage: python scripts/device/parity_device.py build | analyze
"""
import csv
import glob
import os
import re
import shutil
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join("scripts", "guardrail"))
import data_pipeline as dp

P = os.path.join("demo_images", "device_checks", "parity")
ASSETS = os.path.join("android", "app", "src", "main", "assets")
MODELS = {"sickle": "sicklescan_model.tflite", "malaria": "malaria_model.tflite", "guardrail": "guardrail_model.tflite"}


def job_sets():
    A = os.path.join("android", "app", "src", "androidTest", "assets")
    rows = list(csv.DictReader(open("data/guardrail/manifest.csv", newline="", encoding="utf-8")))
    ev = list(csv.DictReader(open("data/guardrail/bbbc_eval.csv", newline="", encoding="utf-8")))
    sets = {}
    sets["25 test assets"] = ([(os.path.join(A, d, n), f"assets/{d}__{n}") for d in ("sample_test_images", "malaria_test_images", "guardrail_test_images")
                               for n in sorted(os.listdir(os.path.join(A, d)))], ["sickle", "malaria", "guardrail"])
    # NOTE (found in Phase 14d): Positive/Unlabelled and Negative/Clear share basenames (both numbered 1.jpg, 2.jpg, ... --
    # 147 collisions). Destination names MUST stay distinct per source folder or the second copy silently overwrites the first.
    sk = ([(p, "positive") for p in sorted(glob.glob("data/raw/Positive/Unlabelled/*.jpg"))]
          + [(p, "negative") for p in sorted(glob.glob("data/raw/Negative/Clear/*.jpg"))])
    sets["sickle images (569)"] = ([(p, f"sickle/{label}__{os.path.basename(p)}") for p, label in sk], ["sickle"])
    nih = [r["path"] for r in rows if r["split"] == "test" and r["source"] == "malaria"]
    sets["NIH single-cell test crops (387)"] = ([(p, "nih/" + os.path.basename(p)) for p in nih], ["malaria", "guardrail"])
    sets["BBBC fields (163)"] = ([(r["path"], "bbbc/" + os.path.basename(r["path"])) for r in ev if r["source"] == "bbbc041_val163"], ["guardrail"])
    sets["BBBC official test images (120)"] = ([(r["path"], "bbbc/" + os.path.basename(r["path"])) for r in ev if r["source"] == "bbbc041_test120"], ["guardrail"])
    held = [r["path"] for r in rows if r["label"] == "not_smear" and r["split"] in ("val", "test") and not r["source"].startswith("synthetic")]
    sets["held-out non-smear (945)"] = ([(p, "neg/" + p.replace("\\", "/").replace("data/guardrail/negatives/", "").replace("/", "__")) for p in held], ["guardrail"])
    fresh = sorted(glob.glob("data/guardrail/fresh_negatives/*/*.jpg"))
    sets["fresh non-smear (986)"] = ([(p, "fresh/" + p.replace("\\", "/").replace("data/guardrail/fresh_negatives/", "").replace("/", "__")) for p in fresh], ["guardrail"])
    return sets


def build():
    shutil.rmtree(P, ignore_errors=True)
    os.makedirs(P)
    z = np.load("artifacts/bbbc041/crops_val.npz", allow_pickle=True)["X"]
    import cv2
    idx = np.random.default_rng(0).choice(len(z), 200, replace=False)
    crops = []
    os.makedirs(os.path.join(P, "crops"))
    for i in idx:
        rel = f"crops/crop{i}.png"
        cv2.imencode(".png", cv2.cvtColor(z[i], cv2.COLOR_RGB2BGR))[1].tofile(os.path.join(P, rel)); crops.append((None, rel))
    jobs, manifest = [], {}
    for name, (files, models) in job_sets().items():
        for src, rel in files:
            os.makedirs(os.path.dirname(os.path.join(P, rel)), exist_ok=True)
            shutil.copy(src, os.path.join(P, rel))
            for m in models:
                jobs.append((rel, m)); manifest[(rel, m)] = name
    for _, rel in crops:
        for m in ("malaria", "guardrail"):
            jobs.append((rel, m)); manifest[(rel, m)] = "BBBC single-cell crops 224x224 (200)"
    with open(os.path.join(P, "jobs.csv"), "w") as f:
        f.write("\n".join(f"{r},{m}" for r, m in jobs) + "\n")
    with open(os.path.join(P, "sets.csv"), "w") as f:
        f.write("\n".join(f"{r},{m},{s}" for (r, m), s in manifest.items()) + "\n")
    print(len(jobs), "jobs;", sum(os.path.getsize(os.path.join(dp_, fn)) for dp_, _, fs in os.walk(P) for fn in fs) / 1e6, "MB")


def status(p, ceiling=65.0):
    conf = (p if p >= .5 else 1 - p) * 100
    return "borderline" if conf < ceiling else ("positive" if p >= .5 else "negative")


def analyze():
    its = {}
    def prob(m, x):
        if m not in its:
            its[m] = tf.lite.Interpreter(model_path=os.path.join(ASSETS, MODELS[m])); its[m].allocate_tensors()
        it = its[m]; i, o = it.get_input_details()[0], it.get_output_details()[0]
        it.set_tensor(i["index"], x[None].astype(np.float32)); it.invoke(); return float(it.get_tensor(o["index"]).reshape(-1)[0])
    sets = {(r, m): s for r, m, s in (l.rstrip("\n").split(",", 2) for l in open(os.path.join(P, "sets.csv")))}
    res = list(csv.DictReader(open(os.path.join(P, "results.csv"))))
    by = {}
    cache = {}
    for r in res:
        rel, m, pd_ = r["path"], r["model"], float(r["P"])
        if rel not in cache:
            cache = {rel: dp.load_and_preprocess_single(os.path.join(P, rel))}
        pt = prob(m, cache[rel])
        by.setdefault((sets[(rel, m)], m), []).append((rel, pt, pd_, float(r["classifyMs"])))
    print(f"{'set / model':58s} {'n':>5s} | |dP| phone-vs-TF mean    max   >0.01 | flips@0.5 | status changes | phone ms/image")
    for (sn, m), v in by.items():
        pt, pd_ = np.array([x[1] for x in v]), np.array([x[2] for x in v]); d = np.abs(pt - pd_)
        chg = sum(status(a) != status(b) for a, b in zip(pt, pd_)) if m in ("sickle", "malaria") else "-"
        print(f"{sn + ' / ' + m:58s} {len(v):5d} | {d.mean():.6f} {d.max():.5f} {int((d > 0.01).sum()):6d} | {int(((pt >= .5) != (pd_ >= .5)).sum()):^9d} | {chg!s:^14s} | {np.mean([x[3] for x in v]):.1f}")
    old = {}
    lg = os.path.join("demo_images", "device_checks", "preproc", "device_log.txt")
    for l in open(lg, encoding="utf-8"):
        mm = re.match(r"(\S+) .* app-preprocessing sickle P=([\d.]+)", l)
        if mm: old[mm[1]] = float(mm[2])
    v = {x[0]: x for x in by[("25 test assets", "sickle")]}
    print(f"\nsickle model on the 25 test assets, phone P: OLD Android resize vs NEW exact resize vs TensorFlow reference")
    for rel, (_, pt, pd_, _) in sorted(v.items()):
        tag = rel.replace("assets/", "").replace("__", "__")
        print(f"  {rel:58s} old {old.get(tag, float('nan')):.4f}  new {pd_:.4f}  TF {pt:.4f}   |new-TF| {abs(pd_ - pt):.4f}")
    t = v["assets/malaria_test_images__parasitized_1.png"]
    print(f"\nphase14c target image (parasitized_1 on the sickle model): NEW on phone {t[2]:.4f} | TensorFlow {t[1]:.4f} | old 0.8515")


if __name__ == "__main__":
    {"build": build, "analyze": analyze}[sys.argv[1]]()
