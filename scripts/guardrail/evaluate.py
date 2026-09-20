"""
Guardrail evaluation + threshold validation + the two post-training checks.

Output convention: sigmoid = P(smear). Reject when P(smear) < REJECT_THRESHOLD.
  * false negative (FN)  = real smear wrongly REJECTED     ("annoying")
  * false positive (FP)  = non-smear wrongly ACCEPTED      ("dangerous": it
                           reaches the disease classifier)

Threshold is chosen on the VALIDATION set only, then reported on the untouched
test set. Rule: among thresholds whose val FN rate <= FN_BUDGET, pick the one
minimizing val FP rate (ties -> middle of the tied range, i.e. largest margin).

Check 1 (shortcut learning): compares P(smear) on real malaria crops against
probes that hold CONTENT constant while flipping the suspected shortcut cue
(black background / small size / low resolution). See run_shortcut_probes().

Check 2: FP rate broken out by negative source (incl. each synthetic subtype),
with Wilson 95% intervals -- per-source test counts are small, so a bare
percentage would overstate precision.

Usage: python scripts/guardrail/evaluate.py --head dense128|simple [--final]
  --final also writes model_output/guardrail/guardrail_results.md
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import IMG_SIZE, load_split, normalize

FN_BUDGET = 0.01
OUT_DIR = os.path.join("model_output", "guardrail")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def predict(model, X):
    Xn = normalize(X).astype(np.float32)
    return model.predict(Xn, batch_size=64, verbose=0).reshape(-1)


def rates(y, p, t):
    """Returns tn, fp, fn, tp with 'smear' as the positive class, accept iff p >= t."""
    pred = p >= t
    tp = int(np.sum(pred & (y == 1)))
    fn = int(np.sum(~pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    return tn, fp, fn, tp


def choose_threshold(y, p):
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
    table = []
    for t in grid:
        tn, fp, fn, tp = rates(y, p, t)
        table.append((float(t), fn / max(fn + tp, 1), fp / max(fp + tn, 1)))
    ok = [r for r in table if r[1] <= FN_BUDGET]
    best_fpr = min(r[2] for r in ok)
    tied = [r[0] for r in ok if abs(r[2] - best_fpr) < 1e-12]
    return float(np.median(tied)), table, (min(tied), max(tied))


def load_gray_free(path):
    return tf.image.resize(
        tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False), [IMG_SIZE, IMG_SIZE]
    ).numpy()


def run_shortcut_probes(model, threshold, Xte, yte, ste):
    """Hold CONTENT fixed, flip the suspected cue, see whether P(smear) follows the cue."""
    rng = np.random.default_rng(0)
    res = {}

    def stats(p):
        return dict(n=int(len(p)), median=float(np.median(p)), p10=float(np.percentile(p, 10)),
                    p90=float(np.percentile(p, 90)), accept_rate=float(np.mean(p >= threshold)))

    mal = Xte[ste == "malaria"]
    sick = Xte[ste == "sickle_cell"]
    neg_real = Xte[(yte == 0) & ~np.char.startswith(ste.astype(str), "synthetic")]

    # (a) baseline confidences by group
    res["A. real malaria test crops (as-is)"] = stats(predict(model, mal))
    res["A. real sickle-cell test fields (as-is)"] = stats(predict(model, sick))
    res["A. real non-smear test images (as-is)"] = stats(predict(model, neg_real))

    def paste_black(X, scale):
        out = np.zeros_like(X)
        s = int(IMG_SIZE * scale)
        small = tf.image.resize(X, [s, s]).numpy()
        o = (IMG_SIZE - s) // 2
        out[:, o:o + s, o:o + s] = small
        return out

    # (b) SAME non-smear content, shrunk onto a black background (malaria-like framing)
    res["B. non-smear content shrunk onto BLACK bg (scale 0.6)"] = stats(predict(model, paste_black(neg_real, 0.6)))
    # (c) SAME malaria content, black background recoloured (cue removed, cell untouched)
    mal_recolor = mal.copy()
    for i in range(len(mal_recolor)):
        m = mal_recolor[i].max(axis=-1, keepdims=True) < 25
        mal_recolor[i] = np.where(m, rng.uniform(0, 255, (1, 1, 3)), mal_recolor[i])
    res["C. malaria crops, black bg RECOLOURED to random colour"] = stats(predict(model, mal_recolor))
    # (d) SAME malaria content, shrunk on black canvas at 0.6 (cue kept, size changed)
    res["D. malaria crops shrunk onto black canvas (scale 0.6)"] = stats(predict(model, paste_black(mal, 0.6)))
    # (e) resolution: SAME non-smear content degraded to ~malaria's native ~124px then upscaled back
    def degrade(X, s=64):
        return tf.image.resize(tf.image.resize(X, [s, s]), [IMG_SIZE, IMG_SIZE]).numpy()
    res["E. non-smear content degraded to 64px (malaria-like blur)"] = stats(predict(model, degrade(neg_real)))
    res["E'. sickle-cell fields degraded to 64px"] = stats(predict(model, degrade(sick)))
    # (f) both cues at once on non-smear content: small + black bg + blurred
    res["F. non-smear: small + black bg + 64px blur (ALL malaria cues)"] = stats(
        predict(model, degrade(paste_black(neg_real, 0.6))))

    # --- probes using transforms the model was NOT trained against (B-F above reuse the
    # training augmentations, so they mostly show the augmentation worked, not independent
    # evidence). G-I mimic what sickle-cell fields actually look like (round microscope
    # field, pink/purple stain colour) applied to NON-smear content.
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    circle = (((xx - IMG_SIZE / 2) ** 2 + (yy - IMG_SIZE / 2) ** 2) <= (IMG_SIZE * 0.48) ** 2)[None, :, :, None]

    def round_field(X):
        return X * circle

    def stain_tint(X):
        return np.clip(X * np.array([1.0, 0.72, 0.92]) + np.array([20, 0, 12]), 0, 255)

    res["G. non-smear content in ROUND microscope-style field (unseen)"] = stats(predict(model, round_field(neg_real)))
    res["H. non-smear content with pink/purple STAIN tint (unseen)"] = stats(predict(model, stain_tint(neg_real)))
    res["I. non-smear: round field + stain tint (unseen)"] = stats(predict(model, round_field(stain_tint(neg_real))))
    res["J. non-smear (synthetic blank/finger) with stain tint (unseen)"] = stats(predict(
        model, stain_tint(Xte[np.isin(ste, ["synthetic_blank", "synthetic_finger", "synthetic_black", "synthetic_blur"])])))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    art = os.path.join("artifacts", "guardrail", args.head)
    model = tf.keras.models.load_model(os.path.join(art, "guardrail_keras.keras"))

    Xva, yva, sva, _ = load_split("val", "float32")
    Xte, yte, ste, pte_paths = load_split("test", "float32")
    pva, pte = predict(model, Xva), predict(model, Xte)

    thr, table, tied_range = choose_threshold(yva, pva)
    print(f"[{args.head}] chosen reject threshold: P(smear) < {thr:.2f} -> reject "
          f"(tied range {tied_range[0]:.2f}-{tied_range[1]:.2f}, FN budget {FN_BUDGET:.0%} on val)")

    out = dict(head=args.head, threshold=thr, tied_range=tied_range, val_sweep=table)
    for name, t in (("t0.5", 0.5), ("chosen", thr)):
        tn, fp, fn, tp = rates(yte, pte, t)
        out[name] = dict(threshold=t, tn=tn, fp=fp, fn=fn, tp=tp,
                         accuracy=(tp + tn) / len(yte), fnr=fn / (fn + tp), fpr=fp / (fp + tn))
        print(f"  test @T={t:.2f}: acc={out[name]['accuracy']:.4f} FNR={out[name]['fnr']:.4f} "
              f"({fn}/{fn + tp}) FPR={out[name]['fpr']:.4f} ({fp}/{fp + tn})")

    # per-source breakdown at the chosen threshold
    per_source = {}
    for s in sorted(set(ste)):
        m = ste == s
        y0 = yte[m][0]
        accepted = int(np.sum(pte[m] >= thr))
        n = int(m.sum())
        k = accepted if y0 == 0 else n - accepted  # errors: FP for negatives, FN for smears
        lo, hi = wilson(k, n)
        per_source[s] = dict(kind="FP" if y0 == 0 else "FN", errors=k, n=n, rate=k / n, ci=(lo, hi),
                             median_p=float(np.median(pte[m])), max_p=float(np.max(pte[m])),
                             min_p=float(np.min(pte[m])))
        print(f"  {s:>18s} {per_source[s]['kind']} {k:>3d}/{n:<3d} = {k / n:6.2%}  "
              f"95%CI [{lo:.1%}, {hi:.1%}]  median P(smear)={np.median(pte[m]):.3f}")
    out["per_source"] = per_source

    # how saturated is the output? (a threshold is only meaningful if some predictions sit between 0 and 1)
    band = (pte > 0.02) & (pte < 0.98)
    out["saturation"] = dict(n_test=int(len(pte)), n_between_0p02_0p98=int(band.sum()),
                             n_between_0p10_0p90=int(((pte > 0.10) & (pte < 0.90)).sum()),
                             val_between_0p02_0p98=int(((pva > 0.02) & (pva < 0.98)).sum()))
    print("\nSaturation:", out["saturation"])

    print("\nShortcut probes (content fixed, cue flipped):")
    probes = run_shortcut_probes(model, thr, Xte, yte, ste)
    for k, v in probes.items():
        print(f"  {k:<62s} n={v['n']:>3d} median={v['median']:.3f} p10={v['p10']:.3f} "
              f"accept={v['accept_rate']:.1%}")
    out["probes"] = probes

    # worst offenders
    fp_idx = np.where((yte == 0) & (pte >= thr))[0]
    fn_idx = np.where((yte == 1) & (pte < thr))[0]
    out["fp_examples"] = [(str(pte_paths[i]), float(pte[i])) for i in fp_idx[np.argsort(-pte[fp_idx])][:15]]
    out["fn_examples"] = [(str(pte_paths[i]), float(pte[i])) for i in fn_idx[np.argsort(pte[fn_idx])][:15]]

    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, "eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez(os.path.join(art, "test_probs.npz"), p=pte, y=yte, s=ste, paths=pte_paths,
             pval=pva, yval=yva, sval=sva)
    print("saved", os.path.join(art, "eval.json"))


if __name__ == "__main__":
    main()
