"""Where do the phone's resize and Python's resize differ? Detailed look at the target image (malaria_test_images__parasitized_1.png)."""
import os, struct
import cv2, numpy as np, tensorflow as tf
D = os.path.join("demo_images", "device_checks", "preproc")
tag = "malaria_test_images__parasitized_1.png"
t_dev = np.fromfile(os.path.join(D, "dev", tag + ".tensor"), "<f4").reshape(224, 224, 3) * 127.5 + 127.5
t_py = np.fromfile(os.path.join(D, tag + ".pytensor"), "<f4").reshape(224, 224, 3) * 127.5 + 127.5
raw = np.load(os.path.join(D, tag + ".pydecoded.npz"))["rgb"]
print("source", raw.shape[:2][::-1], "-> 224x224 (scale x%.3f horizontally, x%.3f vertically)" % (224 / raw.shape[1], 224 / raw.shape[0]))
print("phone tensor values are whole 0-255 numbers:", bool(np.allclose(t_dev, np.rint(t_dev), atol=1e-3)), "| python tensor whole numbers:", bool(np.allclose(t_py, np.rint(t_py), atol=1e-3)))
d = (t_dev - t_py)
print("signed mean diff (phone - python): %.3f  | mean abs %.3f | 99th pct abs %.2f | max abs %.1f" % (d.mean(), np.abs(d).mean(), np.percentile(np.abs(d), 99), np.abs(d).max()))
a = np.abs(d).mean(-1)
print("mean abs diff by region: outer 8px border %.3f | interior %.3f" % ((np.concatenate([a[:8].ravel(), a[-8:].ravel(), a[:, :8].ravel(), a[:, -8:].ravel()])).mean(), a[8:-8, 8:-8].mean()))
# compare with the local image gradient: differences should be large where the image has strong edges (sampling-position difference)
g = np.abs(cv2.Sobel(cv2.cvtColor(t_py.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(cv2.cvtColor(t_py.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F, 0, 1))
print("correlation between |phone-python| and local edge strength: %.2f" % np.corrcoef(a.ravel(), g.ravel())[0, 1])
# is the phone tensor consistent with a small SPATIAL SHIFT of the python resize? (sampling-position difference)
best = None
for dx in np.arange(-0.6, 0.61, 0.1):
    for dy in np.arange(-0.6, 0.61, 0.1):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        s = cv2.warpAffine(t_py, M, (224, 224), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        e = np.abs(s - t_dev)[8:-8, 8:-8].mean()
        if best is None or e < best[0]:
            best = (e, dx, dy)
print("best spatial shift of the python tensor to match the phone (interior mean abs err %.3f): dx=%.1f px, dy=%.1f px (at 224 resolution); unshifted err %.3f" % (best[0], best[1], best[2], np.abs(t_py - t_dev)[8:-8, 8:-8].mean()))
