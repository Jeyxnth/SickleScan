"""Phase 14b: what the bundled v3 guardrail (android asset) gives on the images the on-device tests use. Python TFLite, app preprocessing."""
import csv, glob, os, sys
import numpy as np, tensorflow as tf
sys.path.insert(0, os.path.join("scripts", "guardrail"))
import data_pipeline as dp
it = tf.lite.Interpreter(model_path=os.path.join("android", "app", "src", "main", "assets", "guardrail_model.tflite")); it.allocate_tensors()
i, o = it.get_input_details()[0], it.get_output_details()[0]
def p(path):
    it.set_tensor(i["index"], dp.load_and_preprocess_single(path)[None].astype(np.float32)); it.invoke()
    return float(it.get_tensor(o["index"]).reshape(-1)[0])
A = "android/app/src/androidTest/assets/"
acc = ["sample_test_images/neg_140.jpg", "sample_test_images/neg_46.jpg", "sample_test_images/pos_20.jpg", "sample_test_images/pos_21.jpg",
       "malaria_test_images/parasitized_1.png", "malaria_test_images/parasitized_2.png", "malaria_test_images/parasitized_3.png", "malaria_test_images/uninfected_1.png"]
rej = ["guardrail_test_images/nonsmear_" + n + ".jpg" for n in ("coco", "places", "lfw", "screenshots", "documents", "synthetic_finger")]
print("must ACCEPT:", [round(p(A + x), 3) for x in acc])
print("must REJECT:", [round(p(A + x), 3) for x in rej])
f = sorted(glob.glob("demo_images/wide_field_eval/*.png"))
pf = [p(x) for x in f]
print(f"20 on-device wide-field fields: accepted {sum(v >= .5 for v in pf)}/{len(pf)}, min P(smear) {min(pf):.3f}")
d = sorted(glob.glob("demo_images/malaria/*.png")) + sorted(glob.glob("demo_images/malaria/*.jpg"))
if d:
    pd_ = [p(x) for x in d]; print(f"{len(d)} local BBBC single-cell demo crops: accepted {sum(v >= .5 for v in pd_)}/{len(d)}")
