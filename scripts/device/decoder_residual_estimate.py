"""Expected on-device residual after the resize fix, from the phone's OWN decoded pixels (saved in Phase 14c): model(TF-exact resize of phone-decoded pixels) vs model(TF path on TF-decoded pixels).
Whatever difference remains is Android's JPEG/PNG decoder, not the resize. (Kotlin's resize == TF's to ~1e-5, verified separately.)"""
import glob, os, struct, sys
import numpy as np, tensorflow as tf
sys.path.insert(0, os.path.join("scripts", "guardrail"))
import data_pipeline as dp
D = os.path.join("demo_images", "device_checks", "preproc")
A = "android/app/src/main/assets/"
its = {n: tf.lite.Interpreter(model_path=A + f) for n, f in (("sickle", "sicklescan_model.tflite"), ("malaria", "malaria_model.tflite"), ("guardrail", "guardrail_model.tflite"))}
for it in its.values(): it.allocate_tensors()
def P(n, x):
    it = its[n]; i, o = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(i["index"], x[None].astype(np.float32)); it.invoke(); return float(it.get_tensor(o["index"]).reshape(-1)[0])
def read_decoded(path):
    b = open(path, "rb").read(); w, h = struct.unpack("<ii", b[:8]); a = np.frombuffer(b[8:], "<u4").reshape(h, w)
    return np.stack([(a >> 16) & 255, (a >> 8) & 255, a & 255], -1).astype(np.float32)
rows = []
for f in sorted(glob.glob(os.path.join(D, "dev", "*.decoded"))):
    tag = os.path.basename(f)[:-8]
    x_dev = dp.normalize(tf.image.resize(read_decoded(f), [224, 224]).numpy())      # phone decode + exact resize
    x_ref = np.fromfile(os.path.join(D, tag + ".pytensor"), "<f4").reshape(224, 224, 3)   # TF decode + TF resize
    rows.append((tag, {n: (P(n, x_ref), P(n, x_dev)) for n in its}, tag.endswith(".png")))
for n in its:
    for kind, sel in (("PNG", True), ("JPEG", False)):
        d = np.array([abs(r[1][n][0] - r[1][n][1]) for r in rows if r[2] == sel])
        fl = sum((r[1][n][0] >= .5) != (r[1][n][1] >= .5) for r in rows if r[2] == sel)
        print(f"{n:9s} {kind:4s} n={len(d):2d}: expected phone-vs-TensorFlow |dP| mean {d.mean():.5f} max {d.max():.5f}, decision flips {fl}")
t = [r for r in rows if r[0].endswith("parasitized_1.png")][0]
print(f"parasitized_1 on the sickle model, expected on the phone after the fix: {t[1]['sickle'][1]:.4f} (TensorFlow {t[1]['sickle'][0]:.4f}, old phone 0.8515)")
