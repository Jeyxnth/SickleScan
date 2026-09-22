package com.sicklescan.app

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Runs the app's real inference path (ImageClassifier, loading the bundled
 * sicklescan_model.tflite exactly as MainActivity does) on a real
 * device/emulator, and checks it reproduces the model's known predictions
 * for 11 held-out test images: the original 8 from Phase 2 (all correctly
 * predicted in Phase 1 evaluation, see /model_output/results.md) plus 3
 * more held-out images added in Phase 3, whose expected predictions were
 * independently computed by running the same .tflite file + identical
 * bilinear-resize/normalize preprocessing through Python/TensorFlow.
 *
 * This must run on a device/emulator (Run > instrumented test in Android
 * Studio, or `./gradlew connectedDebugAndroidTest`) because the TFLite
 * native library only runs under the real Android runtime/ABI — it cannot
 * be exercised in a plain desktop JVM unit test.
 */
@RunWith(AndroidJUnit4::class)
class ImageClassifierInstrumentedTest {

    // fileName -> expected predicted label ("positive"/"negative").
    private val expected = mapOf(
        // Original 8 (Phase 2), all correct vs. ground truth in Phase 1.
        "pos_325.jpg" to "positive",
        "pos_231.jpg" to "positive",
        "pos_20.jpg" to "positive",
        "pos_64.jpg" to "positive",
        "pos_213.jpg" to "positive",
        "pos_155.jpg" to "positive",
        "pos_95.jpg" to "positive",
        "neg_46.jpg" to "negative",
        // 3 new held-out images (Phase 3), also correct vs. ground truth.
        "pos_21.jpg" to "positive",   // moderate confidence (~0.73) -- exercises the borderline path
        "neg_140.jpg" to "negative",
        "pos_393.jpg" to "positive",
    )

    @Test
    fun matchesKnownPredictions() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val classifier = ImageClassifier(context, Disease.SICKLE_CELL)

        try {
            expected.forEach { (fileName, expectedLabel) ->
                val bitmap = InstrumentationRegistry.getInstrumentation().context.assets.open("sample_test_images/$fileName").use { stream ->
                    BitmapFactory.decodeStream(stream)
                }
                requireNotNull(bitmap) { "Failed to decode test asset $fileName" }

                val positiveProbability = classifier.classify(bitmap)
                val predictedLabel = if (positiveProbability >= 0.5f) "positive" else "negative"

                assertEquals(
                    "Prediction mismatch for $fileName (p=$positiveProbability)",
                    expectedLabel,
                    predictedLabel
                )
            }
        } finally {
            classifier.close()
        }
    }
}
