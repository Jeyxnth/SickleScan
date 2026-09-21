"""
Convert the BBBC041 malaria classifier to TFLite and verify it, same approach as the earlier models:
build int8-dynamic / float16 / float32, compare each against the Keras model on
  * the full official BBBC041 TEST crops (5,917), the VAL crops (1,823),
  * the four real user photos (whole image resized to 224x224 bilinear, as the app does),
and only then pick. Writes model_output/malaria_bbbc041/{malaria_bbbc041_model.tflite, labels.txt}.
"""
import glob
import json
import os

import cv2
import numpy as np
import tensorflow as tf

ART = os.path.join("artifacts", "bbbc041")
OUT = os.path.join("model_output", "malaria_bbbc041")
PHOTOS = sorted(p for p in glob.glob(os.path.join("data", "tiling_validation", "malaria_positive_fields", "*"))
                if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))


def photo_input(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def build(saved, kind):
    conv = tf.lite.TFLiteConverter.from_saved_model(saved)
    if kind == "int8_dynamic":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    elif kind == "float16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    return conv.convert()


def run(blob, X):
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    out = []
    for x in X:
        it.set_tensor(i["index"], ((x[None].astype(np.float32) - 127.5) / 127.5))
        it.invoke()
        out.append(it.get_tensor(o["index"]).reshape(-1)[0])
    return np.array(out)


if __name__ == "__main__":
    model = tf.keras.models.load_model(os.path.join(ART, "model", "bbbc_keras.keras"))
    saved = os.path.join(ART, "model", "saved_model")
    model.export(saved)   # avoids the TF2.16/Keras3 from_keras_model MLIR bug

    sets = {}
    for s in ("test", "val"):
        z = np.load(os.path.join(ART, f"crops_{s}.npz"), allow_pickle=True)
        sets[s] = (z["X"].astype(np.float32), z["y"])
    sets["photos"] = (np.stack([photo_input(p) for p in PHOTOS]), None)
    keras_p = {k: model.predict((X - 127.5) / 127.5, batch_size=64, verbose=0).reshape(-1) for k, (X, _) in sets.items()}

    blobs, results = {}, {}
    for kind in ("int8_dynamic", "float16", "float32"):
        blobs[kind] = build(saved, kind)
        r = dict(size_kb=len(blobs[kind]) / 1024)
        for k, (X, y) in sets.items():
            p = run(blobs[kind], X)
            kp = keras_p[k]
            r[k] = dict(label_agreement=float(np.mean((p >= .5) == (kp >= .5))), max_abs_diff=float(np.max(np.abs(p - kp))),
                        mean_abs_diff=float(np.mean(np.abs(p - kp))))
            if y is not None:
                r[k]["accuracy"] = float(np.mean((p >= .5) == (y == 1)))
                r[k]["sensitivity"] = float(np.mean(p[y == 1] >= .5))
                r[k]["specificity"] = float(np.mean(p[y == 0] < .5))
            if k == "photos":
                r[k]["per_photo"] = {os.path.basename(pp): (round(float(kp[i]), 4), round(float(p[i]), 4)) for i, pp in enumerate(PHOTOS)}
        results[kind] = r
        print(kind, json.dumps(r, indent=1), flush=True)

    f16 = results["float16"]
    ok = all(f16[k]["label_agreement"] >= 0.999 for k in ("test", "val")) and f16["photos"]["label_agreement"] == 1.0
    chosen = "float16" if ok else "float32"
    print("chosen:", chosen)
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "malaria_bbbc041_model.tflite"), "wb").write(blobs[chosen])
    open(os.path.join(OUT, "labels.txt"), "w", encoding="utf-8").write("uninfected\nparasitized\n")
    json.dump(dict(chosen=chosen, results=results), open(os.path.join(ART, "quant_comparison.json"), "w"), indent=1)
    print("saved", os.path.join(OUT, "malaria_bbbc041_model.tflite"), f"{len(blobs[chosen]) / 1024:.0f} KB")
