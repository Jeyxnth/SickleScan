package com.sicklescan.app

import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.abs

/**
 * On-device check of the wide-field malaria pipeline against the Python TFLite pipeline (Phase 13), plus real latency.
 *
 * Needs the evaluation set built by scripts/detector/make_device_eval_set.py (BBBC041 VALIDATION fields; not in the repo
 * because BBBC041 is CC BY-NC-SA). Push it once:
 *   adb push demo_images/wide_field_eval /sdcard/Android/data/com.sicklescan.app/files/wide_field_eval
 * then run:  ./gradlew connectedDebugAndroidTest --tests "*WideFieldPipelineInstrumentedTest"
 * and read the results with:  adb logcat -d -s WideFieldEval
 * Without that folder the test is skipped (not failed). The reserved user photos are never used here.
 */
@RunWith(AndroidJUnit4::class)
class WideFieldPipelineInstrumentedTest {

    private class Expected(val file: String, val label: String, val cells: Int, val topScore: Float)

    @Test
    fun deviceMatchesPythonPipelineAndReportsLatency() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = File(ctx.getExternalFilesDir(null), "wide_field_eval")
        assumeTrue("evaluation set not pushed to ${dir.path}", File(dir, "expected.csv").exists())
        val expected = File(dir, "expected.csv").readLines().drop(1).filter { it.isNotBlank() }.map {
            val p = it.split(",")
            Expected(p[0], p[1], p[2].toInt(), p[3].toFloat())
        }

        val pipeline = WideFieldMalariaPipeline(ctx)
        try {
            // The app calls warmUp() as soon as wide-field malaria is chosen, before Analyze: time it, then the first real analysis.
            val tWarm = System.nanoTime()
            pipeline.warmUp()
            Log.i(TAG, "warmUp(): ${(System.nanoTime() - tWarm) / 1_000_000} ms")
            val first = expected.first()
            val firstRun = pipeline.analyse(BitmapFactory.decodeFile(File(dir, first.file).path))
            Log.i(TAG, "FIRST analysis after warmUp(): ${firstRun.timings.totalMs} ms (${firstRun.field.cellsAnalysed} cells)")

            var decisionMismatches = 0
            var bigScoreDiffs = 0
            var cellCountDiffs = 0
            val totals = ArrayList<Long>()
            val perStage = LongArray(4)
            for (e in expected) {
                val bmp = BitmapFactory.decodeFile(File(dir, e.file).path)
                val out = pipeline.analyse(bmp)
                val t = out.timings
                totals += t.totalMs
                perStage[0] += t.prepareMs; perStage[1] += t.detectMs; perStage[2] += t.cropMs; perStage[3] += t.classifyMs
                // The full answer (Positive / Negative / Inconclusive from the 20-cell count gate) must match what the Python
                // pipeline gives for this field: same top score and cell count -> same answer via the shared rule.
                val expectedStatus = WideFieldMalaria.interpret(FloatArray(e.cells) { if (it == 0) e.topScore else 0f }).status
                if (out.field.status != expectedStatus) decisionMismatches++
                if (abs(out.field.topCellScore - e.topScore) > 0.05f) bigScoreDiffs++
                if (abs(out.field.cellsAnalysed - e.cells) > 2) cellCountDiffs++
                Log.i(
                    TAG,
                    "${e.file} [${e.label}] cells ${out.field.cellsAnalysed} (py ${e.cells}) top ${"%.4f".format(out.field.topCellScore)} " +
                        "(py ${"%.4f".format(e.topScore)}) total ${t.totalMs} ms = prep ${t.prepareMs} + detect ${t.detectMs} + crop ${t.cropMs} + classify ${t.classifyMs}",
                )
                bmp.recycle()
            }
            val sorted = totals.sorted()
            Log.i(
                TAG,
                "SUMMARY ${expected.size} fields: median ${sorted[sorted.size / 2]} ms, mean ${totals.average().toLong()} ms, max ${sorted.last()} ms; " +
                    "mean per stage: prepare ${perStage[0] / expected.size}, detect ${perStage[1] / expected.size}, " +
                    "crop ${perStage[2] / expected.size}, classify ${perStage[3] / expected.size} ms | " +
                    "decision mismatches vs Python: $decisionMismatches, top-score differences > 0.05: $bigScoreDiffs, cell counts differing by > 2: $cellCountDiffs",
            )
            assertEquals("answers (incl. the count gate) must match the Python pipeline", 0, decisionMismatches)
            assertTrue("too many top-score differences vs Python: $bigScoreDiffs of ${expected.size}", bigScoreDiffs <= expected.size / 10)
        } finally {
            pipeline.close()
        }
    }

    companion object {
        private const val TAG = "WideFieldEval"
    }
}
