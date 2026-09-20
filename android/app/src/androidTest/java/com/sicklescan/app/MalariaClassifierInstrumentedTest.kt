package com.sicklescan.app

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Runs the app's real inference path (ImageClassifier(context, Disease.MALARIA),
 * loading the bundled malaria_model.tflite exactly as the app does) on a
 * real device/emulator, and checks it reproduces the model's actual
 * predictions -- independently computed by running the same .tflite file
 * + identical bilinear-resize/normalize preprocessing through Python
 * against these same 8 images.
 *
 * `parasitized_misclassified.png` is deliberately a real error case (the
 * model itself predicts "uninfected" for a truly parasitized cell, ~3.1%
 * of test images do this per malaria_results.md) -- kept rather than
 * cherry-picked away, so this test verifies the Kotlin implementation
 * faithfully reproduces the model's actual behavior, errors included, not
 * that the model is always right.
 *
 * Must run on a device/emulator (`./gradlew connectedDebugAndroidTest`) --
 * the TFLite native library only runs under the real Android runtime/ABI.
 */
@RunWith(AndroidJUnit4::class)
class MalariaClassifierInstrumentedTest {

    // fileName -> expected predicted label ("parasitized"/"uninfected"),
    // verified against model_output/malaria/malaria_model.tflite directly.
    private val expected = mapOf(
        "uninfected_1.png" to "uninfected",
        "parasitized_misclassified.png" to "uninfected", // real model error; true label is parasitized
        "uninfected_2.png" to "uninfected",
        "parasitized_1.png" to "parasitized",
        "uninfected_3.png" to "uninfected",
        "uninfected_4.png" to "uninfected",
        "parasitized_2.png" to "parasitized",
        "parasitized_3.png" to "parasitized",
    )

    @Test
    fun matchesKnownPredictions() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val classifier = ImageClassifier(context, Disease.MALARIA)

        try {
            expected.forEach { (fileName, expectedLabel) ->
                val bitmap = context.assets.open("malaria_test_images/$fileName").use { stream ->
                    BitmapFactory.decodeStream(stream)
                }
                requireNotNull(bitmap) { "Failed to decode test asset $fileName" }

                val positiveProbability = classifier.classify(bitmap)
                val predictedLabel = if (positiveProbability >= 0.5f) "parasitized" else "uninfected"

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
