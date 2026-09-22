"""Golden values for ImageOpsTest.tfBilinear...: tf.image.resize (default bilinear, the call the classifiers were trained/validated with) on the same deterministic
synthetic image formula as make_kotlin_goldens.py. Writes android/app/src/test/resources/goldens/tf_resize.txt (32x32x3 subsample of the float result per case)."""
import os
import numpy as np
import tensorflow as tf

OUT = os.path.join("android", "app", "src", "test", "resources", "goldens", "tf_resize.txt")


def synth(w, h):
    y, x = np.mgrid[0:h, 0:w]
    return np.stack([(x * 7 + y * 13 + ((x * y) % 37) * 5 + c * 50) % 256 for c in range(3)], -1).astype(np.uint8)


lines = []
for (w, h) in ((53, 41), (127, 169), (97, 136), (300, 260), (640, 600), (1000, 999)):
    r = tf.image.resize(synth(w, h).astype(np.float32), [224, 224]).numpy()
    lines.append(f"case {w} {h}")
    lines.append(" ".join(f"{v:.5f}" for v in r[::7, ::7].reshape(-1)))
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote", len(lines) // 2, "cases")
