package com.sicklescan.app

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Phase 8: ONE decoded image goes through the guardrail and then BOTH disease
 * models, exactly as ScreenFragment does when both conditions are selected.
 * Expected probabilities were computed in Python by running the same bundled
 * .tflite files with identical preprocessing (all three models take a float32
 * [1,224,224,3] input, verified). Tolerance allows for device/JPEG-decoder drift.
 *
 * Note the second row: the sickle-cell model, given a NIH-style malaria cell image, still
 * outputs a confident "positive" (0.81), while the (BBBC041-trained, Phase 10) malaria model
 * calls that same NIH-style cell uninfected (0.03). The disease models are only meaningful on
 * their own image type; this test pins the behaviour, it doesn't endorse it.
 *
 * Must run on a device/emulator (`./gradlew connectedDebugAndroidTest`); not run here.
 */
@RunWith(AndroidJUnit4::class)
class SharedImageBothDiseasesInstrumentedTest {

    // asset -> (sickle P(pos), malaria P(pos))
    private val expected = mapOf(
        "sample_test_images/pos_20.jpg" to (0.9646f to 0.0945f),
        "malaria_test_images/parasitized_1.png" to (0.8061f to 0.0288f),
    )

    @Test
    fun sameImageThroughGuardrailAndBothDiseaseModels() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val guardrail = ImageClassifier(context, "guardrail_model.tflite", "guardrail_labels.txt")
        val sickle = ImageClassifier(context, Disease.SICKLE_CELL)
        val malaria = ImageClassifier(context, Disease.MALARIA)
        try {
            expected.forEach { (asset, want) ->
                val bitmap = InstrumentationRegistry.getInstrumentation().context.assets.open(asset).use { BitmapFactory.decodeStream(it) }
                requireNotNull(bitmap) { "Failed to decode $asset" }

                assertTrue("guardrail should accept $asset", GuardrailInterpreter.looksLikeSmear(guardrail.classify(bitmap)))
                assertEquals("sickle P(pos) for $asset", want.first, sickle.classify(bitmap), 0.03f)
                assertEquals("malaria P(pos) for $asset", want.second, malaria.classify(bitmap), 0.03f)
            }
        } finally {
            guardrail.close(); sickle.close(); malaria.close()
        }
    }
}
