"""Re-estimate BatchNorm moving statistics of a trained detector from training-set crops (weights untouched).
Usage: python scripts/detector/recalibrate_bn.py in.keras out.keras [n_batches]"""
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
import train as T
from common import splits

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    m = tf.keras.models.load_model(src, custom_objects={"loss_fn": T.loss_fn}, compile=False)
    sp = splits()
    data = T.Data(np.load(os.path.join("artifacts", "detector", "imgs_train.npy"), mmap_mode="r"), sp["train"], n, True, 777)
    bns = [l for l in m.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    for l in bns:  # exponential average of batch statistics; 0.95**200 ~ 3.5e-5 so the reset values carry no weight
        l.momentum = 0.95
    # reset running stats
    for l in bns:
        l.moving_mean.assign(tf.zeros_like(l.moving_mean)); l.moving_variance.assign(tf.ones_like(l.moving_variance))
    for i in range(n):
        m(data[i][0], training=True)
    print("recalibrated", len(bns), "BN layers over", n, "batches")
    m.save(dst)
