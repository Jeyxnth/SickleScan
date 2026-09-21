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
 * Phase 10: the bundled malaria model is now the BBBC041-trained classifier (native-scale
 * thin-smear cell crops), NOT the NIH-trained one these images were bundled for. These 8 images
 * are NIH-style segmented cells on a black background; the new model calls ALL of them
 * "uninfected", including the four truly parasitized ones (P(infected) 0.0001-0.03). That is a
 * real limitation of the new model on NIH-style crops, recorded here rather than hidden: this
 * test only verifies that the Kotlin inference path reproduces the model's actual output
 * (independently computed in Python from the same .tflite with the same preprocessing), not that
 * the model is right on these images. Correctness on BBBC041-style crops is validated in
 * model_output/malaria_bbbc041/ (the BBBC-derived demo crops are not bundled here: their
 * redistribution licence is unverified).
 *
 * Must run on a device/emulator (`./gradlew connectedDebugAndroidTest`) --
 * the TFLite native library only runs under the real Android runtime/ABI.
 */
@RunWith(AndroidJUnit4::class)
class MalariaClassifierInstrumentedTest {

    // fileName -> label the SHIPPED (BBBC041) model actually predicts, verified in Python
    // against the bundled malaria_model.tflite. Parasitized ones are misses -- see note above.
    private val expected = mapOf(
        "uninfected_1.png" to "uninfected",
        "parasitized_misclassified.png" to "uninfected", // true label parasitized; model miss
        "uninfected_2.png" to "uninfected",
        "parasitized_1.png" to "uninfected",             // true label parasitized; model miss (P=0.029)
        "uninfected_3.png" to "uninfected",
        "uninfected_4.png" to "uninfected",
        "parasitized_2.png" to "uninfected",             // true label parasitized; model miss (P=0.002)
        "parasitized_3.png" to "uninfected",             // true label parasitized; model miss (P=0.0001)
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
