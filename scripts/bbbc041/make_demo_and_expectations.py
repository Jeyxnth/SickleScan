"""
(1) Build a demo set of single-cell crop images from BBBC041 VAL crops (held out from training; random, seeded)
    into demo_images/malaria/ (gitignored: BBBC041 image licence not verified for redistribution).
(2) Run the SHIPPED malaria .tflite (android assets) with the app's preprocessing on: the demo crops, the 8 NIH
    test images already bundled for instrumented tests, and the sickle/malaria sample pair used by the
    both-conditions tests, printing the expected probabilities used to update those tests.
"""
import csv
import glob
import os
import random
import sys

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join("scripts", "guardrail"))
from data_pipeline import load_and_preprocess_single

ART = os.path.join("artifacts", "bbbc041")
DEMO = os.path.join("demo_images", "malaria")
ASSETS = os.path.join("android", "app", "src", "main", "assets")
TEST_ASSETS = os.path.join("android", "app", "src", "androidTest", "assets")


def interp(name):
    it = tf.lite.Interpreter(model_path=os.path.join(ASSETS, name))
    it.allocate_tensors()
    return it


def prob(it, x):
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(i["index"], x[None].astype(np.float32))
    it.invoke()
    return float(it.get_tensor(o["index"]).reshape(-1)[0])


if __name__ == "__main__":
    os.makedirs(DEMO, exist_ok=True)
    z = np.load(os.path.join(ART, "crops_val.npz"), allow_pickle=True)
    X, y, cats = z["X"], z["y"], z["cats"]
    rng = random.Random(11)
    inf_idx = rng.sample([i for i in range(len(y)) if y[i] == 1], 6)
    neg_idx = rng.sample([i for i in range(len(y)) if cats[i] == "red blood cell"], 6)
    mal = interp("malaria_model.tflite")
    rows = []
    for tag, idxs in (("infected", inf_idx), ("uninfected", neg_idx)):
        for n, i in enumerate(idxs, 1):
            path = os.path.join(DEMO, f"{tag}_{n}_{cats[i].replace(' ', '')}.png")
            cv2.imwrite(path, cv2.cvtColor(X[i], cv2.COLOR_RGB2BGR))
            p = prob(mal, load_and_preprocess_single(path))
            rows.append((os.path.basename(path), tag, cats[i], p))
    with open(os.path.join(DEMO, "expected.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "true_label", "bbbc_category", "shipped_model_P_infected"])
        w.writerows([(a, b, c, f"{p:.4f}") for a, b, c, p in rows])
    print("DEMO CROPS (val, held out from training) -> shipped model P(infected):")
    for a, b, c, p in rows:
        ok = (p >= .5) == (b == "infected")
        print(f"  {a:34s} true={b:10s} P={p:.4f} {'OK' if ok else 'WRONG'}  confidence={max(p, 1 - p) * 100:.1f}%")
    acc = np.mean([(p >= .5) == (b == "infected") for _, b, _, p in rows])
    print(f"  demo-set accuracy {acc:.0%} ({sum((p >= .5) == (b == 'infected') for _, b, _, p in rows)}/{len(rows)})")

    print("\nNIH test images already bundled for instrumented tests (shipped BBBC model):")
    for f in sorted(glob.glob(os.path.join(TEST_ASSETS, "malaria_test_images", "*"))):
        print(f"  {os.path.basename(f):32s} P(infected)={prob(mal, load_and_preprocess_single(f)):.4f}")

    print("\nPair for both-conditions tests (sickle P, NEW malaria P):")
    sick = interp("sicklescan_model.tflite")
    for f in (os.path.join(TEST_ASSETS, "sample_test_images", "pos_20.jpg"), os.path.join(TEST_ASSETS, "malaria_test_images", "parasitized_1.png")):
        x = load_and_preprocess_single(f)
        print(f"  {os.path.basename(f):22s} sickle P(pos)={prob(sick, x):.4f}  malaria P(pos)={prob(mal, x):.4f}")
