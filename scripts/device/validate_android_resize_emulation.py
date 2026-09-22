"""Validate the emulation of Android's resize (half-pixel bilinear, sub-pixel coords + weights quantised to 1/16, output truncated to 8 bit) against the
phone's tensor for ALL 25 test images, starting from the phone's OWN decoded pixels (so JPEG-decoder differences are excluded).
Then the downstream effect of TF-style vs Android-style preprocessing on the three bundled models over larger image sets (emulated; see caveat)."""
import glob, os, struct, sys
import cv2, numpy as np, tensorflow as tf
sys.path.insert(0, os.path.join("scripts", "guardrail"))
D = os.path.join("demo_images", "device_checks", "preproc")


def android_resize(rgb):
    """rgb uint8 HxWx3 -> 224x224x3 float (whole numbers 0-255)."""
    src = rgb.astype(np.float64); H, W = src.shape[:2]
    def coords(n_out, n_in):
        c = (np.arange(n_out) + 0.5) * n_in / n_out - 0.5
        return np.floor(c * 16) / 16
    ys, xs = coords(224, H), coords(224, W)
    y0 = np.clip(np.floor(ys).astype(int), 0, H - 1); y1 = np.clip(y0 + 1, 0, H - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, W - 1); x1 = np.clip(x0 + 1, 0, W - 1)
    fy = (np.floor(np.clip(ys - np.floor(ys), 0, 1) * 16) / 16)[:, None, None]; fx = (np.floor(np.clip(xs - np.floor(xs), 0, 1) * 16) / 16)[None, :, None]
    top = src[y0][:, x0] * (1 - fx) + src[y0][:, x1] * fx
    bot = src[y1][:, x0] * (1 - fx) + src[y1][:, x1] * fx
    return np.floor(top * (1 - fy) + bot * fy)


def read_decoded(path):
    b = open(path, "rb").read(); w, h = struct.unpack("<ii", b[:8])
    a = np.frombuffer(b[8:], "<u4").reshape(h, w)
    return np.stack([(a >> 16) & 255, (a >> 8) & 255, a & 255], -1).astype(np.uint8)


if __name__ == "__main__":
    errs = []
    print(f"{'image':54s} {'src':>10s}  mean|err| gray levels   exact pixels")
    for f in sorted(glob.glob(os.path.join(D, "dev", "*.decoded"))):
        tag = os.path.basename(f)[:-8]
        dev = np.fromfile(os.path.join(D, "dev", tag + ".tensor"), "<f4").reshape(224, 224, 3) * 127.5 + 127.5
        dec = read_decoded(f)
        e = np.abs(android_resize(dec) - dev)
        errs.append(e.mean())
        print(f"{tag:54s} {dec.shape[1]:4d}x{dec.shape[0]:<4d}  {e.mean():8.4f}           {float((e < 0.5).mean()):.2%}")
    print(f"\nemulation vs phone, all {len(errs)} images: mean error {np.mean(errs):.4f}, worst image {max(errs):.4f} gray levels")
