"""Can Android's Bitmap.createScaledBitmap(filter=true) be explained as an 8-bit fixed-point bilinear (truncating, coarse sub-pixel weights)?
Emulate a few variants on the decoded target image and compare with the phone's tensor."""
import os
import numpy as np
D = os.path.join("demo_images", "device_checks", "preproc")
tag = "malaria_test_images__parasitized_1.png"
dev = np.fromfile(os.path.join(D, "dev", tag + ".tensor"), "<f4").reshape(224, 224, 3) * 127.5 + 127.5
src = np.load(os.path.join(D, tag + ".pydecoded.npz"))["rgb"].astype(np.float64)
H, W = src.shape[:2]


def bilinear(fbits, rounding, center):
    # sample position for output pixel o: (o + 0.5) * scale - 0.5 (half-pixel centres) or o * scale (corner aligned)
    def coords(n_out, n_in):
        o = np.arange(n_out)
        c = (o + 0.5) * n_in / n_out - 0.5 if center else o * (n_in - 1) / (n_out - 1)
        if fbits:
            c = np.floor(c * (1 << fbits)) / (1 << fbits)    # sub-pixel position quantised
        return c
    ys, xs = coords(224, H), coords(224, W)
    y0 = np.clip(np.floor(ys).astype(int), 0, H - 1); y1 = np.clip(y0 + 1, 0, H - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, W - 1); x1 = np.clip(x0 + 1, 0, W - 1)
    fy = np.clip(ys - np.floor(ys), 0, 1)[:, None, None]; fx = np.clip(xs - np.floor(xs), 0, 1)[None, :, None]
    if fbits:
        fy = np.floor(fy * (1 << fbits)) / (1 << fbits); fx = np.floor(fx * (1 << fbits)) / (1 << fbits)
    top = src[y0][:, x0] * (1 - fx) + src[y0][:, x1] * fx
    bot = src[y1][:, x0] * (1 - fx) + src[y1][:, x1] * fx
    out = top * (1 - fy) + bot * fy
    return np.floor(out) if rounding == "floor" else np.rint(out) if rounding == "round" else out


print(f"{'variant':60s} mean|err| vs phone (whole image)   exact-match pixels")
for center in (True, False):
    for fbits in (0, 4, 8, 16):
        for rounding in ("float", "round", "floor"):
            e = bilinear(fbits, rounding, center)
            err = np.abs(e - dev)
            print(f"{('half-pixel centres' if center else 'corner aligned'):18s} sub-pixel bits={fbits:2d} output={rounding:5s}    {err.mean():6.3f}                   {float((err < 0.5).mean()):.1%}")
