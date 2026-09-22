"""End-to-end: probability predicted by (phone-decoded pixels -> emulated Android resize -> desktop TFLite) vs the probability the PHONE logged, for every asset image."""
import glob, os, re, sys
import numpy as np, tensorflow as tf
sys.path.insert(0, os.path.join("scripts", "device")); sys.path.insert(0, os.path.join("scripts", "guardrail"))
from validate_android_resize_emulation import android_resize, read_decoded
import data_pipeline as dp
D = os.path.join("demo_images", "device_checks", "preproc")
it = tf.lite.Interpreter(model_path="android/app/src/main/assets/sicklescan_model.tflite"); it.allocate_tensors()
i, o = it.get_input_details()[0], it.get_output_details()[0]
def P(x):
    it.set_tensor(i["index"], x[None].astype(np.float32)); it.invoke(); return float(it.get_tensor(o["index"]).reshape(-1)[0])
dev = {}
for l in open(os.path.join(D, "device_log.txt"), encoding="utf-8"):
    m = re.match(r"(\S+) .* app-preprocessing sickle P=([\d.]+)", l)
    if m: dev[m[1]] = float(m[2])
rows = []
for f in sorted(glob.glob(os.path.join(D, "dev", "*.decoded"))):
    tag = os.path.basename(f)[:-8]
    emu = P(dp.normalize(android_resize(read_decoded(f))))
    tf_ = P(np.fromfile(os.path.join(D, tag + ".pytensor"), "<f4").reshape(224, 224, 3))
    rows.append((tag, dev[tag], emu, tf_))
print(f"{'image':54s} phone(app)  emulation  TF-preproc | |emulation-phone|  |TF-phone|")
for t, d, e, p in rows: print(f"{t:54s} {d:9.4f}  {e:9.4f}  {p:9.4f} | {abs(e - d):.5f}   {abs(p - d):.4f}")
print(f"\nmax |emulation - phone| = {max(abs(e - d) for _, d, e, _ in rows):.5f} ; max |TF - phone| = {max(abs(p - d) for _, d, _, p in rows):.4f}")
