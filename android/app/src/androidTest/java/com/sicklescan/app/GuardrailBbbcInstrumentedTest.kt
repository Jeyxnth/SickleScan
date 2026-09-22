package com.sicklescan.app

import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * On-device check that the retrained guardrail (Phase 14) ACCEPTS BBBC041-style photos: the wide-field validation fields pushed for
 * [WideFieldPipelineInstrumentedTest] (`files/wide_field_eval`) and, if present, BBBC041 single-cell validation crops
 * (`files/guardrail_crops`). Python reference for the same files: 20/20 fields accepted (min P(smear) 0.87), 12/12 crops accepted.
 * Skipped if the folders were not pushed. Read the per-image scores with `adb logcat -d -s GuardrailBbbc`.
 */
@RunWith(AndroidJUnit4::class)
class GuardrailBbbcInstrumentedTest {

    private fun scoreFolder(name: String, extensions: Set<String>): List<Pair<String, Float>> {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = File(ctx.getExternalFilesDir(null), name)
        val files = dir.listFiles { f -> f.extension.lowercase() in extensions }?.sortedBy { it.name } ?: emptyList()
        assumeTrue("no images pushed to ${dir.path}", files.isNotEmpty())
        val guardrail = ImageClassifier(ctx, "guardrail_model.tflite", "guardrail_labels.txt")
        try {
            return files.map { f ->
                val bmp = BitmapFactory.decodeFile(f.path)
                val p = guardrail.classify(bmp)
                bmp.recycle()
                Log.i(TAG, "$name/${f.name}: P(smear) = ${"%.3f".format(p)}")
                f.name to p
            }
        } finally {
            guardrail.close()
        }
    }

    @Test
    fun wideFieldPhotosAreAccepted() {
        val scores = scoreFolder("wide_field_eval", setOf("png", "jpg"))
        val rejected = scores.filter { !GuardrailInterpreter.looksLikeSmear(it.second) }
        Log.i(TAG, "wide-field: accepted ${scores.size - rejected.size}/${scores.size}, min ${"%.3f".format(scores.minOf { it.second })}")
        assertTrue("wide-field photos wrongly rejected: $rejected", rejected.isEmpty())
    }

    @Test
    fun singleCellCropsAreAccepted() {
        val scores = scoreFolder("guardrail_crops", setOf("png", "jpg"))
        val rejected = scores.filter { !GuardrailInterpreter.looksLikeSmear(it.second) }
        Log.i(TAG, "single-cell: accepted ${scores.size - rejected.size}/${scores.size}, min ${"%.3f".format(scores.minOf { it.second })}")
        assertTrue("single-cell crops wrongly rejected: $rejected", rejected.isEmpty())
    }

    companion object {
        private const val TAG = "GuardrailBbbc"
    }
}
