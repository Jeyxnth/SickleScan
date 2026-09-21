"""
Evaluate the BBBC041 classifier.
 1. Crop-level on the official test set (120 images, different resolution than training) using the
    GROUND-TRUTH boxes as an oracle detector: AUC, sensitivity/specificity at 0.5, per-stage recall.
 2. Val leukocyte false-positive rate (test.json contains no leukocytes).
 3. Image-level with oracle boxes: for each test image, does 'any GT cell >= T' / '>=2 cells >= T' flag it?
    (Every test image contains infected cells, so this measures sensitivity only; false-alarm rate on
    infected-free images is measured in val and on the sickle-dataset control fields in validate_photos.py.)
"""
import collections
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

OUT = os.path.join("artifacts", "bbbc041")


def predict(model, X):
    return model.predict((X.astype(np.float32) - 127.5) / 127.5, batch_size=64, verbose=0).reshape(-1)


if __name__ == "__main__":
    model = tf.keras.models.load_model(os.path.join(OUT, "model", "bbbc_keras.keras"))
    t = np.load(os.path.join(OUT, "crops_test.npz"), allow_pickle=True)
    p, y, cats, imgs = predict(model, t["X"]), t["y"], t["cats"], t["imgs"]
    print(f"TEST crops: {len(y)} ({int(y.sum())} infected)  AUC={roc_auc_score(y, p):.4f}")
    for thr in (0.5, 0.7, 0.9):
        pred = p >= thr
        tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
        fp, tn = int((pred & (y == 0)).sum()), int((~pred & (y == 0)).sum())
        print(f"  thr {thr}: sensitivity {tp / (tp + fn):.3f} ({tp}/{tp + fn})  specificity {tn / (tn + fp):.4f}  FP {fp}  precision {tp / max(tp + fp, 1):.3f}")
    for c in ("ring", "trophozoite", "schizont", "gametocyte"):
        m = cats == c
        print(f"  recall @0.5 for {c:12s}: {np.mean(p[m] >= .5):.3f} (n={int(m.sum())})")

    v = np.load(os.path.join(OUT, "crops_val.npz"), allow_pickle=True)
    pv = predict(model, v["X"])
    for c in ("leukocyte", "red blood cell"):
        m = v["cats"] == c
        print(f"VAL {c}: predicted infected @0.5 = {np.mean(pv[m] >= .5):.3f} (n={int(m.sum())})")
    m = v["y"] == 1
    print(f"VAL infected recall @0.5 = {np.mean(pv[m] >= .5):.3f}; AUC={roc_auc_score(v['y'], pv):.4f}")

    # image-level with oracle boxes
    print("\nIMAGE-LEVEL (oracle boxes), TEST:")
    by_img = collections.defaultdict(list)
    for i, im in enumerate(imgs):
        by_img[im].append(i)
    for name, rule in (("any cell >= 0.5", lambda q: (q >= .5).any()), ("any cell >= 0.9", lambda q: (q >= .9).any()),
                       (">=2 cells >= 0.5", lambda q: (q >= .5).sum() >= 2), (">=3 cells >= 0.5", lambda q: (q >= .5).sum() >= 3)):
        flagged = [rule(p[idx]) for idx in by_img.values()]
        has_inf = [bool(y[idx].max()) for idx in by_img.values()]
        tp = sum(f and h for f, h in zip(flagged, has_inf))
        print(f"  {name:18s}: flags {sum(flagged)}/{len(flagged)} images; images with infected cells caught {tp}/{sum(has_inf)}")
    # image-level false alarms: val images that contain NO infected GT cell (oracle boxes)
    by_v = collections.defaultdict(list)
    for i, im in enumerate(v["imgs"]):
        by_v[im].append(i)
    clean = [idx for idx in by_v.values() if v["y"][idx].max() == 0]
    print(f"VAL images with no infected cell (in the sampled crops): {len(clean)}")
    for name, rule in (("any cell >= 0.5", lambda q: (q >= .5).any()), ("any cell >= 0.9", lambda q: (q >= .9).any()), (">=2 cells >= 0.5", lambda q: (q >= .5).sum() >= 2)):
        if clean:
            print(f"  {name:18s}: false-alarm on {sum(rule(pv[idx]) for idx in clean)}/{len(clean)} clean images (note: RBCs subsampled to <=1500 in val, so this understates per-image cell counts)")
