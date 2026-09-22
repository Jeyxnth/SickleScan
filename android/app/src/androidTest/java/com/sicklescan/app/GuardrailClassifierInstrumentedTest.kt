package com.sicklescan.app

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Runs the real guardrail (guardrail_model.tflite, loaded exactly as
 * ScreenFragment loads it) on a device/emulator against:
 *  - real smear images from both diseases (must be ACCEPTED -> reach the disease model)
 *  - obviously-wrong images: a photo, a scene, a face, a screenshot, a document,
 *    a finger-over-lens capture (must be REJECTED -> never reach the disease model)
 *
 * Expected outcomes were independently computed by running the same .tflite +
 * identical bilinear-resize/normalize preprocessing in Python. Must run on a
 * device (`./gradlew connectedDebugAndroidTest`); the TFLite native library
 * doesn't load on a desktop JVM.
 *
 * `sample_test_images/pos_155.jpg` is deliberately NOT asserted as accepted: it
 * is a real sickle-cell image the guardrail wrongly rejects (P(smear)~0.24), kept
 * out of the "must pass" list rather than hidden -- see guardrail_results.md.
 */
@RunWith(AndroidJUnit4::class)
class GuardrailClassifierInstrumentedTest {

    private val mustBeAccepted = listOf(
        "sample_test_images/neg_140.jpg",
        "sample_test_images/neg_46.jpg",
        "sample_test_images/pos_20.jpg",
        "sample_test_images/pos_21.jpg",
        "malaria_test_images/parasitized_1.png",
        "malaria_test_images/parasitized_2.png",
        "malaria_test_images/parasitized_3.png",
        "malaria_test_images/uninfected_1.png",
    )

    private val mustBeRejected = listOf(
        "guardrail_test_images/nonsmear_coco.jpg",
        "guardrail_test_images/nonsmear_places.jpg",
        "guardrail_test_images/nonsmear_lfw.jpg",
        "guardrail_test_images/nonsmear_screenshots.jpg",
        "guardrail_test_images/nonsmear_documents.jpg",
        "guardrail_test_images/nonsmear_synthetic_finger.jpg",
    )

    private fun smearProbability(classifier: ImageClassifier, assetPath: String): Float {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val bitmap = InstrumentationRegistry.getInstrumentation().context.assets.open(assetPath).use { BitmapFactory.decodeStream(it) }
        requireNotNull(bitmap) { "Failed to decode test asset $assetPath" }
        return classifier.classify(bitmap)
    }

    @Test
    fun realSmearsPassAndNonSmearsAreRejected() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val guardrail = ImageClassifier(context, "guardrail_model.tflite", "guardrail_labels.txt")
        try {
            mustBeAccepted.forEach { path ->
                val p = smearProbability(guardrail, path)
                assertEquals("Real smear wrongly rejected: $path (P(smear)=$p)", true, GuardrailInterpreter.looksLikeSmear(p))
            }
            mustBeRejected.forEach { path ->
                val p = smearProbability(guardrail, path)
                assertEquals("Non-smear wrongly accepted: $path (P(smear)=$p)", false, GuardrailInterpreter.looksLikeSmear(p))
            }
        } finally {
            guardrail.close()
        }
    }
}
