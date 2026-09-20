"""
Guardrail data pipeline. Same preprocessing as the disease models: decode,
resize to 224x224 bilinear, mobilenet_v2 (pixel-127.5)/127.5 normalization.
Sources are mixed JPEG/PNG so decode_image is used (the disease pipelines are
single-format).

Images are decoded/resized once and cached as .npz under artifacts/guardrail/
(train as uint8 so the training augmentations run on a small in-memory array;
val/test as exact float32 so evaluation matches on-device inference).

Training-only augmentation targets the shortcuts identified in the Phase 7
dataset review -- malaria crops are small (~124px) with black backgrounds
while negatives are larger/sharper/no black background:
  * random downscale->upscale blur on ALL images (resolution/sharpness is
    not allowed to be a class cue)
  * black-canvas paste (image shrunk onto a black canvas), applied to
    negatives MORE often than positives, so "small object on black" is
    also seen as not-a-smear
  * background jitter: near-black pixels recolored to a random colour, so
    "black background" is also not a class cue
plus flips / rotation / brightness / contrast as in the disease models.
"""
import csv
import os

import numpy as np
import tensorflow as tf

IMG_SIZE = 224
CLASS_NAMES = ["not_smear", "smear"]  # index 0 / 1; sigmoid output = P(smear)
MANIFEST = os.path.join("data", "guardrail", "manifest.csv")
CACHE_DIR = os.path.join("artifacts", "guardrail")

P_PASTE_NEG, P_PASTE_POS = 0.4, 0.2
P_BG_JITTER = 0.4
P_DOWNSCALE = 0.5


def read_manifest(split):
    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == split:
                rows.append(r)
    return rows


def load_resized(path):
    """Decode + bilinear-resize to 224x224 float32 in [0,255] (pre-normalization)."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return tf.cast(img, tf.float32).numpy()


def normalize(x):
    return (x - 127.5) / 127.5  # identical to mobilenet_v2.preprocess_input


def load_and_preprocess_single(path):
    return normalize(load_resized(path))


def load_split(split, dtype):
    """Returns X (N,224,224,3) in [0,255], y (N,), sources (N,), paths (N,)."""
    cache = os.path.join(CACHE_DIR, f"{split}_{dtype}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"], z["sources"], z["paths"]
    rows = read_manifest(split)
    X = np.stack([load_resized(r["path"]) for r in rows])
    if dtype == "uint8":
        X = np.clip(np.round(X), 0, 255).astype(np.uint8)
    y = np.array([1 if r["label"] == "smear" else 0 for r in rows], dtype=np.float32)
    sources = np.array([r["source"] for r in rows])
    paths = np.array([r["path"] for r in rows])
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, X=X, y=y, sources=sources, paths=paths)
    return X, y, sources, paths


@tf.function
def _augment(img, label):
    img = tf.cast(img, tf.float32)  # [0,255]
    # 1. random paste onto a black canvas (shrinks the image, adds black background)
    p_paste = tf.where(label > 0.5, P_PASTE_POS, P_PASTE_NEG)
    if tf.random.uniform([]) < p_paste:
        s = tf.random.uniform([], 0.35, 0.85)
        size = tf.cast(s * IMG_SIZE, tf.int32)
        small = tf.image.resize(img, [size, size])
        off_y = tf.random.uniform([], 0, IMG_SIZE - size + 1, dtype=tf.int32)
        off_x = tf.random.uniform([], 0, IMG_SIZE - size + 1, dtype=tf.int32)
        img = tf.pad(
            small,
            [[off_y, IMG_SIZE - size - off_y], [off_x, IMG_SIZE - size - off_x], [0, 0]],
        )
    # 1b. round microscope-style field (black outside a circle) -- added after the Phase 7
    # shortcut check showed round black-cornered fields made non-smear frames look like smears.
    # Mostly on negatives; positives get it rarely (sickle fields are already round).
    p_round = tf.where(label > 0.5, 0.1, 0.4)
    if tf.random.uniform([]) < p_round:
        r = tf.random.uniform([], 0.35, 0.5) * IMG_SIZE
        cx = IMG_SIZE / 2 + tf.random.uniform([], -0.08, 0.08) * IMG_SIZE
        cy = IMG_SIZE / 2 + tf.random.uniform([], -0.08, 0.08) * IMG_SIZE
        yy, xx = tf.meshgrid(tf.range(IMG_SIZE, dtype=tf.float32), tf.range(IMG_SIZE, dtype=tf.float32), indexing="ij")
        inside = tf.cast(((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r, tf.float32)[..., None]
        img = img * inside
    # 1c. stain-like colour tint on negatives (pink/purple grading) so colour alone isn't a cue
    if label < 0.5 and tf.random.uniform([]) < 0.3:
        gain = tf.stack([1.0, tf.random.uniform([], 0.6, 1.0), tf.random.uniform([], 0.75, 1.05)])
        img = tf.clip_by_value(img * gain, 0.0, 255.0)
    # 2. background jitter: near-black pixels -> random colour
    if tf.random.uniform([]) < P_BG_JITTER:
        mask = tf.reduce_max(img, axis=-1, keepdims=True) < 25.0
        color = tf.random.uniform([1, 1, 3], 0.0, 255.0)
        img = tf.where(mask, color, img)
    # 3. random downscale -> upscale (destroys resolution/sharpness cues)
    if tf.random.uniform([]) < P_DOWNSCALE:
        s = tf.cast(tf.random.uniform([], 24.0, 224.0), tf.int32)
        small = tf.image.resize(img, [s, s])
        img = tf.image.resize(small, [IMG_SIZE, IMG_SIZE])
    img = tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, 3])
    # 4. geometric / photometric, as in the disease models
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    img = tf.image.random_brightness(img, 0.15 * 255.0)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return (img - 127.5) / 127.5, label


def make_train_dataset(X, y, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(len(X), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_eval_dataset(X, y, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((normalize(X).astype(np.float32), y))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
