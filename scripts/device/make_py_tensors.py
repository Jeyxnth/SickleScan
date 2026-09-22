"""Phase 14c investigation: for every androidTest asset image, write the tensor the ORIGINAL Python path produces
(scripts/guardrail/data_pipeline.load_and_preprocess_single: tf.io.decode_image + tf.image.resize bilinear + (x-127.5)/127.5, i.e. what
_pair_probs.py fed the models when the expected values were computed) as <folder>__<name>.pytensor (little-endian float32), to push to the phone.
Also writes the Python-decoded RGB pixels (<folder>__<name>.pydecoded) for a pixel-level comparison with what BitmapFactory decoded.
Usage: python scripts/device/make_py_tensors.py OUTDIR"""
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join("scripts", "guardrail"))
import data_pipeline as dp

A = os.path.join("android", "app", "src", "androidTest", "assets")
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
n = 0
for folder in ("sample_test_images", "malaria_test_images", "guardrail_test_images"):
    for name in sorted(os.listdir(os.path.join(A, folder))):
        p = os.path.join(A, folder, name)
        x = dp.load_and_preprocess_single(p).astype("<f4")
        x.tofile(os.path.join(out, f"{folder}__{name}.pytensor"))
        raw = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False).numpy()
        np.savez_compressed(os.path.join(out, f"{folder}__{name}.pydecoded.npz"), rgb=raw)
        n += 1
print(n, "images")
