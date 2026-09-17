package com.sicklescan.app

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.IOException

/**
 * Wraps the SickleScan TFLite model. Preprocessing here must match training
 * exactly: images were resized to the model's expected input size with
 * bilinear interpolation, then scaled from [0,255] to [-1,1] via
 * `(pixel - 127.5) / 127.5` (this is Keras' `mobilenet_v2.preprocess_input`,
 * mode "tf"). The model's actual input tensor shape/dtype (float32,
 * [1, 224, 224, 3], no built-in quantization) was inspected directly with
 * the TFLite Python interpreter before writing this — not assumed.
 */
class ImageClassifier(context: Context) {

    class ClassifierException(message: String, cause: Throwable? = null) : Exception(message, cause)

    private val interpreter: Interpreter
    val labels: List<String>
    private val inputWidth: Int
    private val inputHeight: Int
    private val imageProcessor: ImageProcessor

    init {
        try {
            val modelBuffer = FileUtil.loadMappedFile(context, MODEL_FILENAME)
            interpreter = Interpreter(modelBuffer)
            labels = FileUtil.loadLabels(context, LABELS_FILENAME)

            val inputShape = interpreter.getInputTensor(0).shape() // [1, height, width, 3]
            inputHeight = inputShape[1]
            inputWidth = inputShape[2]

            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(inputHeight, inputWidth, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(NORMALIZE_MEAN, NORMALIZE_STD))
                .build()
        } catch (e: IOException) {
            throw ClassifierException("Failed to load model or labels from assets", e)
        } catch (e: Exception) {
            throw ClassifierException("Failed to initialize the on-device model", e)
        }

        if (labels.size != 2) {
            throw ClassifierException("Expected 2 labels in $LABELS_FILENAME, found ${labels.size}")
        }
    }

    /**
     * Runs inference on [bitmap] and returns the model's raw sigmoid output:
     * the probability that the image is positive (sickle cells present),
     * in [0,1]. Throws [ClassifierException] on failure; never crashes.
     *
     * Deliberately returns just the raw probability rather than a
     * pre-interpreted verdict — see [ScreeningInterpreter] for the
     * confidence/borderline/referral logic built on top of it. There is no
     * per-region or tile-level score here: this model was trained and
     * validated only on whole-field images, so it has no tile-level ground
     * truth to make a sliding-window "severity" score meaningful (see
     * ScreeningInterpreter's file comment).
     */
    fun classify(bitmap: Bitmap): Float {
        try {
            var tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            tensorImage = imageProcessor.process(tensorImage)

            // Single sigmoid output: labels[0] = negative, labels[1] = positive
            // (order fixed by labels.txt / how the model was trained).
            val output = Array(1) { FloatArray(1) }
            interpreter.run(tensorImage.buffer, output)

            return output[0][0]
        } catch (e: Exception) {
            throw ClassifierException("Inference failed", e)
        }
    }

    fun close() {
        interpreter.close()
    }

    companion object {
        private const val MODEL_FILENAME = "sicklescan_model.tflite"
        private const val LABELS_FILENAME = "labels.txt"

        // (x - 127.5) / 127.5 maps [0,255] -> [-1,1], matching
        // tf.keras.applications.mobilenet_v2.preprocess_input used in training.
        private const val NORMALIZE_MEAN = 127.5f
        private const val NORMALIZE_STD = 127.5f
    }
}
