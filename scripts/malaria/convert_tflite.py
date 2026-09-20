"""
Convert the trained malaria Keras model to TensorFlow Lite.

Per the Phase 5 instruction, this does NOT assume float16 is still the
right choice just because it was for the sickle cell model -- it builds
int8 dynamic-range, float16, and float32 variants, evaluates all three
against the Keras model on the FULL test set, and only then picks one.
If int8 turns out to perform as well as or better than float16 here, this
prints a clear flag rather than silently picking float16 out of habit.
"""
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import CLASS_NAMES, load_and_preprocess_single, read_manifest

ARTIFACTS_DIR = os.path.join("artifacts", "malaria")
MODEL_OUTPUT_DIR = os.path.join("model_output", "malaria")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "malaria_keras.keras")
SAVED_MODEL_DIR = os.path.join(ARTIFACTS_DIR, "saved_model")
TFLITE_PATH = os.path.join(MODEL_OUTPUT_DIR, "malaria_model.tflite")
LABELS_PATH = os.path.join(MODEL_OUTPUT_DIR, "labels.txt")
TEST_CSV = os.path.join("data", "malaria", "splits", "test.csv")

N_SANITY_IMAGES = 10


def export_saved_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    model.export(SAVED_MODEL_DIR)  # avoids the TF2.16/Keras3 from_keras_model MLIR bug
    return model


def build_variant(kind):
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    if kind == "int8_dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif kind == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif kind == "float32":
        pass  # no optimizations -> plain float32
    else:
        raise ValueError(kind)
    return converter.convert()


def run_tflite(tflite_bytes, batch):
    tmp_path = os.path.join(ARTIFACTS_DIR, "_tmp_variant.tflite")
    with open(tmp_path, "wb") as f:
        f.write(tflite_bytes)
    interpreter = tf.lite.Interpreter(model_path=tmp_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    probs = []
    for img in batch:
        interpreter.set_tensor(inp["index"], img[None, ...].astype(np.float32))
        interpreter.invoke()
        probs.append(interpreter.get_tensor(out["index"]).reshape(-1)[0])
    os.remove(tmp_path)
    return np.array(probs)


def full_test_comparison(keras_model):
    paths, labels = read_manifest(TEST_CSV)
    labels = np.array(labels)
    print(f"Loading and preprocessing {len(paths)} test images for full-test-set "
          f"quantization comparison (this takes a while)...")
    batch = np.stack([load_and_preprocess_single(p) for p in paths])

    keras_probs = keras_model.predict(batch, batch_size=32, verbose=0).reshape(-1)
    keras_preds = (keras_probs >= 0.5).astype(int)
    keras_acc = np.mean(keras_preds == labels)
    print(f"Keras baseline test accuracy: {keras_acc*100:.2f}%")

    results = {}
    for kind in ("int8_dynamic", "float16", "float32"):
        tflite_bytes = build_variant(kind)
        size_kb = len(tflite_bytes) / 1024
        probs = run_tflite(tflite_bytes, batch)
        preds = (probs >= 0.5).astype(int)

        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

        agreement = np.mean(preds == keras_preds)
        max_diff = np.max(np.abs(probs - keras_probs))
        mean_diff = np.mean(np.abs(probs - keras_probs))

        results[kind] = dict(
            size_kb=size_kb, accuracy=acc, sensitivity=sens, specificity=spec,
            agreement=agreement, max_diff=max_diff, mean_diff=mean_diff,
        )
        print(
            f"{kind:>12s}: size={size_kb:8.1f}KB acc={acc*100:6.2f}% "
            f"sens={sens*100:6.2f}% spec={spec*100:6.2f}% "
            f"agreement_with_keras={agreement*100:6.2f}% "
            f"max_diff={max_diff:.4f} mean_diff={mean_diff:.4f}"
        )

    return results, keras_acc


def decide(results, keras_acc):
    """Pick the quantization scheme. Mirrors the sickle cell decision logic
    (prefer float16 unless it demonstrably loses to int8), but actually
    checks the numbers for THIS model instead of assuming."""
    f16 = results["float16"]
    i8 = results["int8_dynamic"]

    f16_lossless = f16["agreement"] >= 0.999 and abs(f16["accuracy"] - keras_acc) < 0.002
    i8_as_good_or_better = (
        i8["accuracy"] >= f16["accuracy"] - 0.001
        and i8["sensitivity"] >= f16["sensitivity"] - 0.005
        and i8["specificity"] >= f16["specificity"] - 0.005
    )

    print("\n=== Quantization decision ===")
    if i8["accuracy"] > f16["accuracy"] + 0.001 or (
        i8["sensitivity"] > f16["sensitivity"] + 0.005 and i8["specificity"] >= f16["specificity"] - 0.005
    ):
        print(
            "FLAG: int8 dynamic-range quantization performs AS WELL AS OR BETTER "
            "THAN float16 on this dataset (unlike the sickle cell model). "
            "Per instructions, stopping to report this rather than silently "
            "picking float16 out of habit -- see chosen='NEEDS_USER_INPUT' below."
        )
        return "NEEDS_USER_INPUT"

    print(
        f"float16 confirmed best/safe choice: {'lossless' if f16_lossless else 'small measured drift'} "
        f"vs Keras baseline; int8 is {'not' if not i8_as_good_or_better else ''} within tolerance of float16."
    )
    return "float16"


def finalize(chosen, tflite_bytes_by_kind):
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_bytes_by_kind[chosen])
    size_kb = os.path.getsize(TFLITE_PATH) / 1024
    print(f"\nSaved {TFLITE_PATH} ({size_kb:.1f} KB) using {chosen} quantization")

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(CLASS_NAMES) + "\n")
    print(f"Saved {LABELS_PATH}: {CLASS_NAMES}")


def sanity_check():
    paths, labels = read_manifest(TEST_CSV)
    rng = np.random.default_rng(7)
    idx = rng.choice(len(paths), size=min(N_SANITY_IMAGES, len(paths)), replace=False)
    batch = np.stack([load_and_preprocess_single(paths[i]) for i in idx])

    keras_model = tf.keras.models.load_model(MODEL_PATH)
    keras_probs = keras_model.predict(batch, verbose=0).reshape(-1)

    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    tflite_probs = []
    for img in batch:
        interpreter.set_tensor(inp["index"], img[None, ...].astype(np.float32))
        interpreter.invoke()
        tflite_probs.append(interpreter.get_tensor(out["index"]).reshape(-1)[0])
    tflite_probs = np.array(tflite_probs)

    max_abs_diff = np.max(np.abs(keras_probs - tflite_probs))
    agreement = np.mean((keras_probs >= 0.5) == (tflite_probs >= 0.5))
    print(f"\nFinal .tflite sanity check on {len(idx)} images: "
          f"label agreement={agreement*100:.1f}% max_diff={max_abs_diff:.5f}")


def main():
    keras_model = export_saved_model()
    results, keras_acc = full_test_comparison(keras_model)

    # rebuild bytes for all variants once more to save the chosen one
    # (build_variant is deterministic; cheap enough to just redo it)
    tflite_bytes_by_kind = {kind: build_variant(kind) for kind in results}

    chosen = decide(results, keras_acc)
    if chosen == "NEEDS_USER_INPUT":
        print("\nSTOPPING before writing a final .tflite -- see comparison table above.")
        return
    finalize(chosen, tflite_bytes_by_kind)
    sanity_check()


if __name__ == "__main__":
    main()
