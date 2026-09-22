"""Train the single-class cell locator on BBBC041 boxes (CPU). Usage: python scripts/detector/train.py [epochs]"""
import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from common import OUT, STORE_SCALE, STRIDE, build_model, gt_boxes, image_path, read_bgr, splits

CROP = 384
BATCH = 16
SCALE_RANGE = (0.2, 0.6)   # native->input scale: cells ~22-65 px (user photos, upscaled to 640, have ~20-70 px cells)
VAL_SCALE = 0.4


def cache(name, recs):
    p = os.path.join(OUT, f"imgs_{name}.npy")
    if not os.path.exists(p):
        arr = None
        for i, r in enumerate(recs):
            im = read_bgr(image_path(r))
            im = cv2.resize(im, (int(im.shape[1] * STORE_SCALE), int(im.shape[0] * STORE_SCALE)), interpolation=cv2.INTER_AREA)
            if arr is None:
                arr = np.lib.format.open_memmap(p, mode="w+", dtype=np.uint8, shape=(len(recs),) + im.shape)
            arr[i] = im
        arr.flush()
    return np.load(p, mmap_mode="r")


def gaussian(hm, cx, cy, sigma):
    r = int(3 * sigma)
    x0, x1 = max(0, cx - r), min(hm.shape[1], cx + r + 1)
    y0, y1 = max(0, cy - r), min(hm.shape[0], cy + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], g)


def make_sample(img, boxes, is_cell, rng, scale, crop=CROP, augment=True):
    """img: half-native BGR. boxes: native px. Returns input (crop,crop,3) float [-1,1], target (crop/4,crop/4,5)."""
    f = scale / STORE_SCALE
    h, w = img.shape[:2]
    rw, rh = max(crop, int(w * f)), max(crop, int(h * f))
    r = cv2.resize(np.asarray(img), (rw, rh), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
    x0 = rng.integers(0, rw - crop + 1); y0 = rng.integers(0, rh - crop + 1)
    patch = r[y0:y0 + crop, x0:x0 + crop].copy()
    b = boxes[is_cell] * scale - np.array([x0, y0, x0, y0], np.float32)
    if augment:
        if rng.random() < 0.5:
            patch = patch[:, ::-1]; b = np.stack([crop - b[:, 2], b[:, 1], crop - b[:, 0], b[:, 3]], 1)
        if rng.random() < 0.5:
            patch = patch[::-1]; b = np.stack([b[:, 0], crop - b[:, 3], b[:, 2], crop - b[:, 1]], 1)
        if rng.random() < 0.5:  # transpose
            patch = patch.transpose(1, 0, 2); b = b[:, [1, 0, 3, 2]]
        patch = np.ascontiguousarray(patch)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
        hsv[..., 1] *= rng.uniform(0.6, 1.4)
        hsv[..., 2] *= rng.uniform(0.75, 1.25)
        patch = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        if rng.random() < 0.1:
            patch = np.repeat(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)[..., None], 3, 2)
        if rng.random() < 0.25:  # blur (photos from phones / different microscopes)
            k = int(rng.choice([3, 5]))
            patch = cv2.GaussianBlur(patch, (k, k), 0)
    x = (cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5
    g = crop // STRIDE
    t = np.zeros((g, g, 5), np.float32)
    for x0_, y0_, x1_, y1_ in b:
        cx, cy = (x0_ + x1_) / 2 / STRIDE, (y0_ + y1_) / 2 / STRIDE
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < g and 0 <= iy < g):
            continue
        bw, bh = (x1_ - x0_) / STRIDE, (y1_ - y0_) / STRIDE
        gaussian(t[..., 0], ix, iy, max(1.0, (bw + bh) / 2 / 6))
        t[iy, ix, 1:5] = [bw, bh, cx - ix, cy - iy]
    return x, t


class Data(tf.keras.utils.PyDataset):
    def __init__(self, imgs, recs, steps, train, seed):
        super().__init__(workers=8, use_multiprocessing=False, max_queue_size=16)
        self.imgs, self.steps, self.train, self.seed = imgs, steps, train, seed
        self.gt = [gt_boxes(r) for r in recs]

    def __len__(self):
        return self.steps

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i + (0 if self.train else 10 ** 6))
        X, T = [], []
        for _ in range(BATCH):
            k = int(rng.integers(len(self.imgs)))
            s = rng.uniform(*SCALE_RANGE) if self.train else VAL_SCALE
            x, t = make_sample(self.imgs[k], self.gt[k][0], self.gt[k][2], rng, s, augment=self.train)
            X.append(x); T.append(t)
        return np.stack(X), np.stack(T)

    def on_epoch_end(self):
        if self.train:
            self.seed += 1000


def loss_fn(y_true, y_pred):
    hm_t = y_true[..., 0]
    p = tf.clip_by_value(tf.sigmoid(y_pred[..., 0]), 1e-4, 1 - 1e-4)
    pos = tf.cast(hm_t >= 0.9999, tf.float32)
    neg = 1.0 - pos
    pos_loss = -tf.math.log(p) * (1 - p) ** 2 * pos
    neg_loss = -tf.math.log(1 - p) * p ** 2 * (1 - hm_t) ** 4 * neg
    n = tf.maximum(tf.reduce_sum(pos), 1.0)
    focal = (tf.reduce_sum(pos_loss) + tf.reduce_sum(neg_loss)) / n
    m = pos[..., None]
    wh = tf.reduce_sum(tf.abs(y_pred[..., 1:3] - y_true[..., 1:3]) * m) / n
    off = tf.reduce_sum(tf.abs(y_pred[..., 3:5] - y_true[..., 3:5]) * m) / n
    return focal + 0.1 * wh + off


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    os.makedirs(OUT, exist_ok=True)
    sp = splits()
    tr_imgs, va_imgs = cache("train", sp["train"]), cache("val", sp["val"])
    print("cached", tr_imgs.shape, va_imgs.shape, flush=True)
    model, base = build_model((None, None, 3))
    steps = 100
    lr = tf.keras.optimizers.schedules.CosineDecay(1e-3, epochs * steps, alpha=0.02)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss_fn)
    cbs = [tf.keras.callbacks.ModelCheckpoint(os.path.join(OUT, "detector_best.keras"), monitor="val_loss", save_best_only=True),
           tf.keras.callbacks.CSVLogger(os.path.join(OUT, "train_log.csv"))]
    t0 = time.time()
    model.fit(Data(tr_imgs, sp["train"], steps, True, 0), validation_data=Data(va_imgs, sp["val"], 20, False, 0),
              epochs=epochs, callbacks=cbs, verbose=2)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
