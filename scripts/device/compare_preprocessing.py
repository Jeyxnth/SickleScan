"""Phase 14c root-cause analysis: phone preprocessing vs the original Python preprocessing, step by step, for every test asset image.
Inputs (pulled from the phone / made on the PC): demo_images/device_checks/preproc/{dev/*.decoded, dev/*.tensor, *.pydecoded.npz, *.pytensor}.
1) DECODE: BitmapFactory pixels vs tf.io.decode_image pixels.  2) RESIZE+NORMALISE: phone tensor vs Python tensor.  3) Which resampling
reproduces the phone's tensor?  4) Effect on the three bundled models (desktop TFLite interpreter, identical for both tensors).
Usage: python scripts/device/compare_preprocessing.py"""
import glob
import os
import struct
import sys

import cv2
import numpy as np
import tensorflow as tf

D = os.path.join("demo_images", "device_checks", "preproc")
ASSETS = os.path.join("android", "app", "src", "main", "assets")
models = {}
for nm, f in (("sickle", "sicklescan_model.tflite"), ("malaria", "malaria_model.tflite"), ("guardrail", "guardrail_model.tflite")):
    it = tf.lite.Interpreter(model_path=os.path.join(ASSETS, f)); it.allocate_tensors(); models[nm] = it


def prob(nm, x):
    it = models[nm]; i, o = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(i["index"], x[None].astype(np.float32)); it.invoke()
    return float(it.get_tensor(o["index"]).reshape(-1)[0])


def read_decoded(path):
    b = open(path, "rb").read()
    w, h = struct.unpack("<ii", b[:8])
    a = np.frombuffer(b[8:], "<u4").reshape(h, w)
    return np.stack([(a >> 16) & 255, (a >> 8) & 255, a & 255], -1).astype(np.uint8)


def norm(rgb_float):
    return (rgb_float - 127.5) / 127.5


rows = []
for f in sorted(glob.glob(os.path.join(D, "dev", "*.decoded"))):
    tag = os.path.basename(f)[:-len(".decoded")]
    dec_dev = read_decoded(f)
    dec_py = np.load(os.path.join(D, tag + ".pydecoded.npz"))["rgb"]
    t_dev = np.fromfile(os.path.join(D, "dev", tag + ".tensor"), "<f4").reshape(224, 224, 3)
    t_py = np.fromfile(os.path.join(D, tag + ".pytensor"), "<f4").reshape(224, 224, 3)
    same_dec = dec_dev.shape == dec_py.shape and int(np.abs(dec_dev.astype(int) - dec_py.astype(int)).max())
    diff255 = np.abs(t_dev - t_py) * 127.5
    # candidate resamplers applied to the (identical) decoded pixels
    h, w = dec_py.shape[:2]
    cands = {
        "cv2 INTER_LINEAR (uint8 out)": norm(cv2.resize(dec_py, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)),
        "tf.image.resize bilinear, half-pixel, float": t_py,
        "tf.image.resize legacy (no half-pixel)": norm(tf.compat.v1.image.resize_bilinear(dec_py[None].astype(np.float32), [224, 224], half_pixel_centers=False).numpy()[0]),
        "tf.image.resize bilinear then round to uint8": norm(np.rint(tf.image.resize(dec_py.astype(np.float32), [224, 224]).numpy())),
        "cv2 INTER_NEAREST": norm(cv2.resize(dec_py, (224, 224), interpolation=cv2.INTER_NEAREST).astype(np.float32)),
        "cv2 INTER_AREA": norm(cv2.resize(dec_py, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)),
    }
    err = {k: float((np.abs(v - t_dev) * 127.5).mean()) for k, v in cands.items()}
    best = min(err, key=err.get)
    rows.append((tag, (w, h), same_dec, float(diff255.mean()), float(diff255.max()), err, best,
                 {nm: (prob(nm, t_py), prob(nm, t_dev)) for nm in models}))

print(f"{'image':52s} {'src':>9s} decode-diff | tensor diff (0-255 units): mean  max | closest resampler to the PHONE tensor (mean abs err)")
for tag, wh, sd, m, mx, err, best, _ in rows:
    print(f"{tag:52s} {wh[0]:4d}x{wh[1]:<4d} {str(sd):>11s} | {m:6.2f} {mx:6.1f} | {best} ({err[best]:.3f}); tf-half-pixel {err['tf.image.resize bilinear, half-pixel, float']:.3f}")
print("\nDecoded pixels identical (max abs diff 0) for", sum(1 for r in rows if r[2] == 0), "of", len(rows), "images")
big = [r for r in rows if r[3] > 0.5]
print("images whose tensors differ by more than 0.5 gray levels on average:", len(big), "of", len(rows))
print("\nDownstream effect (probabilities: python-tensor vs phone-tensor, same desktop model):")
for nm in ("sickle", "malaria", "guardrail"):
    d = np.array([abs(r[7][nm][0] - r[7][nm][1]) for r in rows])
    print(f"  {nm:9s}: max |dP| {d.max():.4f}, mean {d.mean():.4f}, images with |dP| > 0.03: {int((d > 0.03).sum())}/{len(d)}; decisions (>=0.5) that flip: "
          f"{sum(1 for r in rows if (r[7][nm][0] >= .5) != (r[7][nm][1] >= .5))}")
t = [r for r in rows if r[0].endswith("parasitized_1.png")][0]
print(f"\nthe test image {t[0]}: sickle P python-tensor {t[7]['sickle'][0]:.4f} vs phone-tensor {t[7]['sickle'][1]:.4f}")
print("\nlargest downstream sickle differences:", sorted([(round(abs(r[7]['sickle'][0] - r[7]['sickle'][1]), 4), r[0]) for r in rows], reverse=True)[:5])
