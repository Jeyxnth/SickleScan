"""
Shared pieces for the Phase 11a single-class cell locator (CenterNet-style, anchor-free).

Labels: every red blood cell and every infected cell (ring / trophozoite / schizont / gametocyte) is the single
class "cell". "difficult" boxes are also cells (they are cells, just ambiguous in stage). Leukocytes are NOT cells:
they stay unlabelled, so the detector learns to skip them (the classifier had a 6.2% false-positive rate on WBCs).
Splits are the same as Phase 9b: seed 42, 85/15 by image from training.json; test = official test.json.
"""
import json
import os
import random

import cv2
import numpy as np

ROOT = os.path.join("data", "bbbc041", "raw", "malaria")
OUT = os.path.join("artifacts", "detector")
INFECTED = {"trophozoite", "schizont", "gametocyte", "ring"}
STORE_SCALE = 0.5        # training images cached at half native resolution
EVAL_LONG_SIDE = 640     # inference policy: longer image side resized to 640 px (same rule for every photo)
STRIDE = 4


def splits():
    rng = random.Random(42)
    train_all = json.load(open(os.path.join(ROOT, "training.json")))
    test = json.load(open(os.path.join(ROOT, "test.json")))
    idx = list(range(len(train_all)))
    rng.shuffle(idx)
    n_val = int(round(0.15 * len(idx)))
    val = [train_all[i] for i in idx[:n_val]]
    train = [train_all[i] for i in idx[n_val:]]
    assert not ({r["image"]["pathname"] for r in val} & {r["image"]["pathname"] for r in train})
    return {"train": train, "val": val, "test": test}


def image_path(rec):
    return os.path.join(ROOT, rec["image"]["pathname"].lstrip("/"))


def read_bgr(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def gt_boxes(rec):
    """-> (boxes Nx4 [x0,y0,x1,y1] native px, is_infected N bool, is_cell N bool). Leukocytes: is_cell False."""
    boxes, inf, cell = [], [], []
    for o in rec["objects"]:
        b = o["bounding_box"]
        boxes.append([b["minimum"]["c"], b["minimum"]["r"], b["maximum"]["c"], b["maximum"]["r"]])
        inf.append(o["category"] in INFECTED)
        cell.append(o["category"] != "leukocyte")
    return np.array(boxes, np.float32).reshape(-1, 4), np.array(inf, bool), np.array(cell, bool)


def build_model(input_shape=(None, None, 3)):
    import tensorflow as tf
    from tensorflow.keras import layers as L
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    c2 = base.get_layer("block_2_add").output            # stride 4, 24 ch
    c3 = base.get_layer("block_5_add").output            # stride 8, 32 ch
    c4 = base.get_layer("block_12_add").output           # stride 16, 96 ch
    c5 = base.get_layer("block_16_project_BN").output    # stride 32, 320 ch

    def lat(x, name):
        return L.Conv2D(64, 1, padding="same", name=name)(x)

    p5 = lat(c5, "lat5")
    p4 = L.Add()([lat(c4, "lat4"), L.UpSampling2D(2, interpolation="nearest")(p5)])
    p3 = L.Add()([lat(c3, "lat3"), L.UpSampling2D(2, interpolation="nearest")(p4)])
    p2 = L.Add()([lat(c2, "lat2"), L.UpSampling2D(2, interpolation="nearest")(p3)])
    x = L.Conv2D(64, 3, padding="same", activation="relu", name="neck")(p2)
    hm = L.Conv2D(32, 3, padding="same", activation="relu")(x)
    hm = L.Conv2D(1, 1, name="heat", bias_initializer=tf.keras.initializers.Constant(-4.6))(hm)  # sigmoid applied outside
    reg = L.Conv2D(32, 3, padding="same", activation="relu")(x)
    reg = L.Conv2D(4, 1, name="reg")(reg)  # w, h (px/STRIDE, log-free), dx, dy
    out = L.Concatenate(name="out")([hm, reg])
    return tf.keras.Model(base.input, out), base


def decode(out, score_thr=0.3, max_det=400, stride=STRIDE):
    """out: (H/4, W/4, 5) raw network output for one image (heat logit, w, h, dx, dy). -> boxes Nx4 (input px), scores."""
    heat = 1 / (1 + np.exp(-out[..., 0]))
    # 3x3 peak picking (replaces NMS)
    pad = np.pad(heat, 1, constant_values=0)
    mx = np.max([pad[i:i + heat.shape[0], j:j + heat.shape[1]] for i in range(3) for j in range(3)], axis=0)
    ys, xs = np.where((heat >= mx) & (heat >= score_thr))
    sc = heat[ys, xs]
    order = np.argsort(-sc)[:max_det]
    ys, xs, sc = ys[order], xs[order], sc[order]
    w = np.maximum(out[ys, xs, 1], 1) * stride
    h = np.maximum(out[ys, xs, 2], 1) * stride
    cx = (xs + out[ys, xs, 3]) * stride
    cy = (ys + out[ys, xs, 4]) * stride
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    return boxes.astype(np.float32), sc.astype(np.float32)


def pad_to_multiple(img, m=32):
    h, w = img.shape[:2]
    H, W = -(-h // m) * m, -(-w // m) * m
    if (H, W) == (h, w):
        return img
    return cv2.copyMakeBorder(img, 0, H - h, 0, W - w, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def prep_eval(img_bgr, long_side=EVAL_LONG_SIDE):
    """BGR image -> (float input (1,H,W,3) in [-1,1], scale used). Longer side -> long_side px."""
    s = long_side / max(img_bgr.shape[:2])
    r = cv2.resize(img_bgr, (int(round(img_bgr.shape[1] * s)), int(round(img_bgr.shape[0] * s))),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    r = pad_to_multiple(r)
    x = (cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5
    return x[None], s


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)
