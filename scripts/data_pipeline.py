"""
Shared tf.data pipeline used by train.py, evaluate.py and convert_tflite.py.

Keeping this in one place guarantees train/val/test all get identical
resizing + preprocessing, and that only the training split gets augmentation.
"""
import csv

import tensorflow as tf

IMG_SIZE = 224
CLASS_NAMES = ["negative", "positive"]  # index 0 / 1 -> label used everywhere
LABEL_TO_INT = {name: i for i, name in enumerate(CLASS_NAMES)}


def read_manifest(csv_path):
    paths, labels = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(row["path"])
            labels.append(LABEL_TO_INT[row["label"]])
    return paths, labels


def _load_and_resize(path, label):
    raw = tf.io.read_file(path)
    img = tf.io.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    return img, label


# Augmentation applied to the TRAINING split only. Built as a small
# Sequential of Keras preprocessing layers so it runs on-device as part of
# the tf.data pipeline (and is skipped entirely in inference mode).
def make_augmenter():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.15),  # up to ~54 degrees either way
            tf.keras.layers.RandomBrightness(0.15),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def _preprocess_mobilenet(img, label):
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, label


def load_and_preprocess_single(path):
    """Single-image version of the exact same resize+preprocess pipeline
    used by make_dataset, so ad-hoc scripts (TFLite conversion/sanity
    checks, sample-prediction dumps) see the same pixels the model was
    trained and evaluated on. Returns a (IMG_SIZE, IMG_SIZE, 3) float32 array."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img.numpy()


def make_dataset(csv_path, batch_size=16, augment=False, shuffle=False):
    paths, labels = read_manifest(csv_path)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(_load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        augmenter = make_augmenter()
        ds = ds.map(
            lambda img, label: (augmenter(img, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.map(_preprocess_mobilenet, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, len(paths)
