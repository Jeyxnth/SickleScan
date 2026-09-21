"""
Validate the borderline confidence ceiling for the BBBC041 malaria classifier on ITS OWN held-out data
(same method as the earlier models: accuracy of predictions below vs at/above a confidence cutoff).
confidence = probability of the predicted class (max(p, 1-p)), in percent; a prediction below the ceiling
would be shown as Borderline. Two held-out sets:
  VAL  : held-out images, same imaging setup as training (used for early stopping only)
  TEST : official test.json, different imaging setup -- never used for training or model selection
Also reports the positive-class decision boundary (0.5) behaviour on TEST, because that, not the ceiling,
is where the domain shift bites (infected scores are compressed).
"""
import os

import numpy as np
import tensorflow as tf

OUT = os.path.join("artifacts", "bbbc041")


def predict(model, X):
    return model.predict((X.astype(np.float32) - 127.5) / 127.5, batch_size=64, verbose=0).reshape(-1)


def binned(name, p, y):
    conf = np.where(p >= 0.5, p, 1 - p) * 100
    pred = p >= 0.5
    correct = pred == (y == 1)
    print(f"\n--- {name}: {len(y)} crops ({int(y.sum())} infected), overall accuracy {correct.mean():.2%} ---")
    edges = [50, 55, 60, 65, 70, 80, 90, 99, 100.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            print(f"  confidence {lo:>5.1f}-{min(hi, 100):>5.1f}%: n={int(m.sum()):5d}  accuracy {correct[m].mean():6.1%}")
    for ceil in (55, 60, 65, 70, 75, 80, 90):
        lo, hi = conf < ceil, conf >= ceil
        print(f"  ceiling {ceil}%: below n={int(lo.sum()):4d} ({lo.mean():.1%}) acc {correct[lo].mean() if lo.sum() else float('nan'):.1%} | at/above n={int(hi.sum()):5d} acc {correct[hi].mean():.1%}")
    # what the ceiling does to each true class
    for ceil in (65,):
        for cls, nm in ((1, "infected"), (0, "uninfected")):
            m = y == cls
            print(f"  ceiling {ceil}%: true {nm}: confident-correct {np.mean(correct[m] & (conf[m] >= ceil)):.1%}, "
                  f"borderline {np.mean(conf[m] < ceil):.1%}, confident-WRONG {np.mean(~correct[m] & (conf[m] >= ceil)):.1%}")


if __name__ == "__main__":
    model = tf.keras.models.load_model(os.path.join(OUT, "model", "bbbc_keras.keras"))
    for split in ("val", "test"):
        z = np.load(os.path.join(OUT, f"crops_{split}.npz"), allow_pickle=True)
        binned(split.upper(), predict(model, z["X"]), z["y"])
