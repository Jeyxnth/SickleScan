"""
Phase 14b: full guardrail validation for several models on IDENTICAL images; accept = P(smear) >= 0.5.
SMEAR sets (want ACCEPT): 163 held-out BBBC041 validation fields; 120 official BBBC041 test images (different microscope); held-out sickle-cell
  and NIH single-cell malaria test images; BBBC041 single-cell crops (1,823 validation crops from held-out images; 5,917 official-test crops).
NON-SMEAR sets (want REJECT): Phase 13 sample (360 natural + 240 synthetic; 70% of it was in the guardrail training split), all held-out real
  non-smear (val+test, 945), held-out synthetic (90), 986 FRESH non-smear never used anywhere.
Usage: python scripts/guardrail_v2/evaluate_all.py NAME=PATH [NAME=PATH ...]   (PATH ending .tflite -> TFLite, else Keras)
"""
import csv
import glob
import math
import os
import random
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join("scripts", "guardrail"))
import data_pipeline as dp

NEG_ROOT = os.path.join("data", "guardrail", "negatives")
FRESH = os.path.join("data", "guardrail", "fresh_negatives")


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def load(paths):
    return np.stack([dp.load_resized(p) for p in paths]) if paths else np.zeros((0, 224, 224, 3), np.float32)


def make_model(path):
    if path.endswith(".tflite"):
        it = tf.lite.Interpreter(model_path=path); it.allocate_tensors()
        i, o = it.get_input_details()[0], it.get_output_details()[0]

        def f(X):
            out = []
            for x in dp.normalize(X):
                it.set_tensor(i["index"], x[None].astype(np.float32)); it.invoke()
                out.append(it.get_tensor(o["index"]).reshape(-1)[0])
            return np.array(out)
        return f
    m = tf.keras.models.load_model(path)
    return lambda X: m.predict(dp.normalize(X).astype(np.float32), batch_size=64, verbose=0).reshape(-1) if len(X) else np.zeros(0)


def phase13_sample():
    rng = random.Random(0)
    out = []
    for cat in sorted(os.listdir(NEG_ROOT)):
        files = sorted(glob.glob(os.path.join(NEG_ROOT, cat, "*")))
        files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))]
        out += [(cat, f) for f in rng.sample(files, min(60, len(files)))]
    return out


if __name__ == "__main__":
    specs = [a.split("=", 1) for a in sys.argv[1:]]
    models = {n: make_model(p) for n, p in specs}
    names = list(models)
    rows = list(csv.DictReader(open(os.path.join("data", "guardrail", "manifest.csv"), newline="", encoding="utf-8")))
    ev = list(csv.DictReader(open(os.path.join("data", "guardrail", "bbbc_eval.csv"), newline="", encoding="utf-8")))
    sets = {}
    p163 = [r["path"] for r in ev if r["source"] == "bbbc041_val163"]
    p120 = [r["path"] for r in ev if r["source"] == "bbbc041_test120"]
    sets["BBBC041 held-out validation fields (163)"] = (load(p163), True, p163)
    sets["BBBC041 official test images (120, other microscope)"] = (load(p120), True, p120)
    sk = [r["path"] for r in rows if r["split"] == "test" and r["source"] == "sickle_cell"]
    ml = [r["path"] for r in rows if r["split"] == "test" and r["source"] == "malaria"]
    sets["sickle-cell test images (86, held out)"] = (load(sk), True, sk)
    sets["NIH single-cell malaria test images (387, held out)"] = (load(ml), True, ml)
    for nm, f in (("BBBC041 single-cell VALIDATION crops (1,823)", "crops_val.npz"), ("BBBC041 single-cell official-TEST crops (5,917, other microscope)", "crops_test.npz")):
        z = np.load(os.path.join("artifacts", "bbbc041", f), allow_pickle=True)
        sets[nm] = (z["X"].astype(np.float32), True, None)
    smp = phase13_sample()
    nat = [f for c, f in smp if not c.startswith("synthetic")]
    syn = [f for c, f in smp if c.startswith("synthetic")]
    sets["Phase 13 sample, natural only (360; ~70% were in training)"] = (load(nat), False, nat)
    sets["Phase 13 sample, synthetic capture failures (240)"] = (load(syn), False, syn)
    held = [r["path"] for r in rows if r["label"] == "not_smear" and r["split"] in ("val", "test") and not r["source"].startswith("synthetic")]
    sets["HELD-OUT real non-smear (guardrail val+test, 945)"] = (load(held), False, held)
    fresh = sorted(glob.glob(os.path.join(FRESH, "*", "*.jpg")))
    sets[f"FRESH unseen non-smear ({len(fresh)})"] = (load(fresh), False, fresh)

    scores = {n: {sn: m(X) for sn, (X, _, _) in sets.items()} for n, m in models.items()}
    w = 34
    print(f"{'set':66s} {'want':>6s} | " + " | ".join(f"{n:{w}s}" for n in names))
    for sn, (X, acc, _) in sets.items():
        cells = []
        for n in names:
            p = scores[n][sn]; k = int((p >= .5).sum()) if acc else int((p < .5).sum()); N = len(p)
            lo, hi = wilson(k, N)
            cells.append(f"{k}/{N} = {k / N:6.1%} [{lo:.1%}-{hi:.1%}]")
        print(f"{sn:66s} {'accept' if acc else 'reject':>6s} | " + " | ".join(f"{c:{w}s}" for c in cells))

    un = [sn for sn in sets if sn.startswith("HELD-OUT") or sn.startswith("FRESH")]
    tot = sum(len(sets[s][0]) for s in un)
    print(f"\nFALSE ACCEPTS on all UNSEEN non-smear (held-out real + fresh = {tot}): " + ", ".join(f"{n} {sum(int((scores[n][s] >= .5).sum()) for s in un)} ({sum(int((scores[n][s] >= .5).sum()) for s in un) / tot:.2%})" for n in names))
    print("\nFALSE ACCEPTS by the last model listed (P(smear) per model):")
    last = names[-1]
    for sn in un:
        for i in np.where(scores[last][sn] >= .5)[0]:
            print(f"  {sets[sn][2][i]}  " + "  ".join(f"{n}={scores[n][sn][i]:.3f}" for n in names))
    np.savez(os.path.join("artifacts", "guardrail_eval_all.npz"), **{f"{n}|{sn}": v for n, d in scores.items() for sn, v in d.items()})
