"""
Step 8-9 of Phase 1: convert the trained Keras model to TensorFlow Lite,
verify the converted model's predictions match the original Keras model
on test images, and write final deliverables to model_output/:
  - sicklescan_model.tflite
  - labels.txt
(results.md and sample_predictions/ are written by evaluate.py)

Quantization: float16, not int8 dynamic-range. Dynamic-range quantization
was tried first (per the original plan, for the smallest file size) but the
sanity check below caught it measurably degrading predictions on the test
set (accuracy 94.2% -> 91.9%, specificity 95.45% -> 86.36%; sensitivity was
unaffected). Float16 gave 100% label agreement with the Keras model (zero
measured accuracy loss) at ~4.7MB vs 2.6MB for int8 -- still small enough
for a mobile app, so it was chosen after checking with the user.
"""
import os

import numpy as np
import tensorflow as tf

from data_pipeline import CLASS_NAMES, load_and_preprocess_single, read_manifest

ARTIFACTS_DIR = "artifacts"
MODEL_OUTPUT_DIR = "model_output"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "sicklescan_keras.keras")
SAVED_MODEL_DIR = os.path.join(ARTIFACTS_DIR, "saved_model")
TFLITE_PATH = os.path.join(MODEL_OUTPUT_DIR, "sicklescan_model.tflite")
LABELS_PATH = os.path.join(MODEL_OUTPUT_DIR, "labels.txt")
TEST_CSV = os.path.join("data", "splits", "test.csv")

N_SANITY_IMAGES = 10


def convert():
    model = tf.keras.models.load_model(MODEL_PATH)

    # NOTE: TFLiteConverter.from_keras_model() hits a known TF 2.16 / Keras 3
    # MLIR bug on this functional model ("missing attribute 'value'" /
    # "Failed to infer result type(s)"). Exporting to a plain TF SavedModel
    # first and converting from that path avoids it.
    model.export(SAVED_MODEL_DIR)
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]  # float16 quantization
    tflite_model = converter.convert()

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(TFLITE_PATH) / 1024
    print(f"Saved {TFLITE_PATH} ({size_kb:.1f} KB)")

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(CLASS_NAMES) + "\n")
    print(f"Saved {LABELS_PATH}: {CLASS_NAMES}")

    return model


def load_sanity_images():
    paths, labels = read_manifest(TEST_CSV)
    rng = np.random.default_rng(7)
    idx = rng.choice(len(paths), size=min(N_SANITY_IMAGES, len(paths)), replace=False)

    batch = [load_and_preprocess_single(paths[i]) for i in idx]
    return np.stack(batch), [paths[i] for i in idx], [labels[i] for i in idx]


def verify(keras_model):
    images, paths, labels = load_sanity_images()

    keras_probs = keras_model.predict(images, verbose=0).reshape(-1)

    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    tflite_probs = []
    for img in images:
        interpreter.set_tensor(input_details[0]["index"], img[None, ...].astype(np.float32))
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]["index"]).reshape(-1)[0]
        tflite_probs.append(out)
    tflite_probs = np.array(tflite_probs)

    max_abs_diff = np.max(np.abs(keras_probs - tflite_probs))
    keras_preds = (keras_probs >= 0.5).astype(int)
    tflite_preds = (tflite_probs >= 0.5).astype(int)
    label_matches = int(np.sum(keras_preds == tflite_preds))

    print("\n--- Keras vs TFLite sanity check on", len(images), "test images ---")
    for p, kp, tp, lbl in zip(paths, keras_probs, tflite_probs, labels):
        print(
            f"  {os.path.basename(p):>12s}  actual={CLASS_NAMES[lbl]:<8s}  "
            f"keras={kp:.4f}  tflite={tp:.4f}  diff={abs(kp-tp):.5f}"
        )
    print(f"Max absolute probability difference: {max_abs_diff:.5f}")
    print(f"Predicted-label agreement: {label_matches}/{len(images)}")

    if label_matches == len(images) and max_abs_diff < 0.05:
        print("PASS: TFLite model predictions match the Keras model.")
    else:
        print(
            "WARNING: TFLite predictions diverge from the Keras model more than "
            "expected — investigate before shipping."
        )


def main():
    keras_model = convert()
    verify(keras_model)


if __name__ == "__main__":
    main()
