"""Estimate (not measured -- old on-device dump doesn't cover this image) what the OLD Android resize would have scored on
data/raw/Positive/Unlabelled/306.jpg, using the validated resize emulation from Phase 14c on TF-decoded pixels."""
import os, sys
import numpy as np, tensorflow as tf
sys.path.insert(0, os.path.join("scripts", "guardrail")); sys.path.insert(0, os.path.join("scripts", "device"))
from validate_android_resize_emulation import android_resize
import data_pipeline as dp
p = "data/raw/Positive/Unlabelled/306.jpg"
raw = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False).numpy()
old_est = dp.normalize(android_resize(raw))
it = tf.lite.Interpreter(model_path="android/app/src/main/assets/sicklescan_model.tflite"); it.allocate_tensors()
i, o = it.get_input_details()[0], it.get_output_details()[0]
it.set_tensor(i["index"], old_est[None].astype(np.float32)); it.invoke()
print(f"306.jpg estimated OLD-Android-resize probability (TF-decoded pixels, not phone-decoded -- estimate only): {it.get_tensor(o['index']).reshape(-1)[0]:.4f}")
print("(TensorFlow reference 0.5220, new phone-measured 0.4765)")
