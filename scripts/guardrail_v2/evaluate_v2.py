"""
Phase 14 validation: old guardrail (v1, the bundled float16 TFLite) vs retrained guardrail (v2) on the same images, accept = P(smear) >= 0.5.
SMEAR sets (want ACCEPT): the 163 held-out BBBC041 validation fields (never trained on), the 120 official BBBC041 test images
  (different microscope), the guardrail's held-out sickle-cell and NIH malaria-cell test images, and the BBBC041 single-cell validation crops.
NON-SMEAR sets (want REJECT): the Phase 13 sample of 360 (with the train/val/test split each image had in the guardrail data), all held-out
  non-smear images (val+test), 986 FRESH non-smear images never used anywhere, and the synthetic capture failures.
Usage: python scripts/guardrail_v2/evaluate_v2.py [v2_model_path]
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

V1 = os.path.join("android", "app", "src", "main", "assets", "guardrail_model.tflite")
V2 = sys.argv[1] if len(sys.argv) > 1 else os.path.join("artifacts", "guardrail_v2", "dense128", "guardrail_keras.keras")
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


class V1Model:
    def __init__(self):
        self.it = tf.lite.Interpreter(model_path=V1); self.it.allocate_tensors()
        self.i, self.o = self.it.get_input_details()[0], self.it.get_output_details()[0]

    def __call__(self, X):
        out = []
        for x in dp.normalize(X):
            self.it.set_tensor(self.i["index"], x[None].astype(np.float32)); self.it.invoke()
            out.append(self.it.get_tensor(self.o["index"]).reshape(-1)[0])
        return np.array(out)


class V2Model:
    def __init__(self):
        self.m = tf.keras.models.load_model(V2)

    def __call__(self, X):
        return self.m.predict(dp.normalize(X).astype(np.float32), batch_size=64, verbose=0).reshape(-1) if len(X) else np.zeros(0)


def manifest_rows():
    return list(csv.DictReader(open(os.path.join("data", "guardrail", "manifest.csv"), newline="", encoding="utf-8")))


def phase13_360():
    """Exactly the Phase 13 sample (scripts/detector/nonsmear_test.py): seed 0, up to 60 per category, categories sorted."""
    rng = random.Random(0)
    out = []
    for cat in sorted(os.listdir(NEG_ROOT)):
        files = sorted(glob.glob(os.path.join(NEG_ROOT, cat, "*")))
        files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))]
        out += [(cat, f) for f in rng.sample(files, min(60, len(files)))]
    return out


if __name__ == "__main__":
    models = {"v1 (Phase 7, bundled)": V1Model(), "v2 (retrained)": V2Model()}
    rows = manifest_rows()
    by_path = {r["path"].replace("\\", "/"): r for r in rows}
    sets = {}   # name -> (X, want_accept, extra info)

    ev = list(csv.DictReader(open(os.path.join("data", "guardrail", "bbbc_eval.csv"), newline="", encoding="utf-8")))
    p163 = [r["path"] for r in ev if r["source"] == "bbbc041_val163"]
    p120 = [r["path"] for r in ev if r["source"] == "bbbc041_test120"]
    sets["BBBC041 held-out validation fields (163)"] = (load(p163), True, p163)
    sets["BBBC041 official test images (120, other microscope)"] = (load(p120), True, p120)
    sk = [r["path"] for r in rows if r["split"] == "test" and r["source"] == "sickle_cell"]
    ml = [r["path"] for r in rows if r["split"] == "test" and r["source"] == "malaria"]
    sets["sickle-cell test images (held out)"] = (load(sk), True, sk)
    sets["NIH single-cell malaria test images (held out)"] = (load(ml), True, ml)
    z = np.load(os.path.join("artifacts", "bbbc041", "crops_val.npz"), allow_pickle=True)
    sets["BBBC041 single-cell validation crops (1,823)"] = (z["X"].astype(np.float32), True, None)

    p360 = phase13_360()
    x360 = load([f for _, f in p360])
    sets["Phase 13 non-smear sample (360)"] = (x360, False, [f for _, f in p360])
    held = [r["path"] for r in rows if r["label"] == "not_smear" and r["split"] in ("val", "test") and not r["source"].startswith("synthetic")]
    synth = [r["path"] for r in rows if r["label"] == "not_smear" and r["source"].startswith("synthetic") and r["split"] in ("val", "test")]
    sets["all HELD-OUT real non-smear (guardrail val+test)"] = (load(held), False, held)
    sets["held-out synthetic capture failures (val+test)"] = (load(synth), False, synth)
    fresh = sorted(glob.glob(os.path.join(FRESH, "*", "*.jpg")))
    sets[f"FRESH unseen non-smear ({len(fresh)})"] = (load(fresh), False, fresh)

    scores = {mn: {sn: m(X) for sn, (X, _, _) in sets.items()} for mn, m in models.items()}
    print(f"{'set':60s} {'n':>5s} | " + " | ".join(f"{mn:38s}" for mn in models))
    for sn, (X, want_accept, _) in sets.items():
        cells = []
        for mn in models:
            p = scores[mn][sn]
            acc = int((p >= .5).sum()); n = len(p)
            k = acc if want_accept else n - acc
            lo, hi = wilson(k, n)
            cells.append(f"{'accepted' if want_accept else 'REJECTED'} {k}/{n} = {k / n:6.1%} [{lo:.1%}-{hi:.1%}]")
        print(f"{sn:60s} {len(X):5d} | " + " | ".join(f"{c:38s}" for c in cells))

    # Phase 13 sample: by category and by the split each image had in the guardrail data
    print("\nPhase 13 sample (360): REJECTED counts by category / by guardrail data split")
    cats = np.array([c for c, _ in p360])
    splits = np.array([by_path.get(f.replace("\\", "/"), {"split": "?"})["split"] for _, f in p360])
    for mn in models:
        p = scores[mn]["Phase 13 non-smear sample (360)"]
        rej = p < .5
        print(f"  {mn}: " + ", ".join(f"{c} {int(rej[cats == c].sum())}/{int((cats == c).sum())}" for c in sorted(set(cats))))
        print(f"    by split: " + ", ".join(f"{s} {int(rej[splits == s].sum())}/{int((splits == s).sum())}" for s in ("train", "val", "test")))
    # false accepts of v2 on non-smear sets, for inspection
    print("\nv2 FALSE ACCEPTS (non-smear accepted, P(smear) >= 0.5):")
    for sn, (X, want_accept, paths) in sets.items():
        if want_accept or paths is None:
            continue
        p = scores["v2 (retrained)"][sn]
        for i in np.where(p >= .5)[0]:
            print(f"  [{sn[:38]}] {paths[i]}  P(smear)={p[i]:.3f}   (v1: {scores['v1 (Phase 7, bundled)'][sn][i]:.3f})")
    print("\nP(smear) on the 163 BBBC fields: v1 median %.3f, v2 median %.3f; v2 min %.3f" % (
        np.median(scores["v1 (Phase 7, bundled)"]["BBBC041 held-out validation fields (163)"]),
        np.median(scores["v2 (retrained)"]["BBBC041 held-out validation fields (163)"]),
        scores["v2 (retrained)"]["BBBC041 held-out validation fields (163)"].min()))
    np.savez(os.path.join("artifacts", "guardrail_v2", "eval_scores.npz"),
             **{f"{mn.split()[0]}|{sn}": v for mn, d in scores.items() for sn, v in d.items()})
