package com.sicklescan.app

import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Speed-knob benchmark for the wide-field pipeline (Phase 13 Task 4): the same fields under several (threads, batch size,
 * crop threads) settings. None of these settings may change a score, so the top score per field is compared with the
 * unbatched, single-threaded-crop reference run and any difference is logged. Same evaluation set as
 * [WideFieldPipelineInstrumentedTest]; read results with `adb logcat -d -s WideFieldBench`. Skipped if the set is missing.
 */
@RunWith(AndroidJUnit4::class)
class WideFieldBenchmarkInstrumentedTest {

    @Test
    fun compareSpeedSettings() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = File(ctx.getExternalFilesDir(null), "wide_field_eval")
        assumeTrue("evaluation set not pushed", File(dir, "expected.csv").exists())
        // every second field of the set: a mix of infected/negative and small/large cell counts
        val files = File(dir, "expected.csv").readLines().drop(1).filter { it.isNotBlank() }.map { it.split(",")[0] }
            .filterIndexed { i, _ -> i % 2 == 0 }
        val configs = listOf(
            "reference   threads4 batch1  cropThreads1" to WideFieldMalariaPipeline.Config(4, 1, 1),
            "batch8      threads4 batch8  cropThreads1" to WideFieldMalariaPipeline.Config(4, 8, 1),
            "batch8+crop4 threads4 batch8 cropThreads4" to WideFieldMalariaPipeline.Config(4, 8, 4),
            "batch16+crop4 threads4 batch16 cropThreads4" to WideFieldMalariaPipeline.Config(4, 16, 4),
            "batch8+crop4 threads6 batch8 cropThreads4" to WideFieldMalariaPipeline.Config(6, 8, 4),
            "batch8+crop6 threads4 batch8 cropThreads6" to WideFieldMalariaPipeline.Config(4, 8, 6),
        )
        var reference: List<Float>? = null
        for ((name, cfg) in configs) {
            val p = WideFieldMalariaPipeline(ctx, cfg)
            try {
                p.analyse(BitmapFactory.decodeFile(File(dir, files.first()).path)) // warm-up
                val tops = ArrayList<Float>()
                val totals = ArrayList<Long>()
                var crop = 0L; var classify = 0L; var detect = 0L
                for (f in files) {
                    val bmp = BitmapFactory.decodeFile(File(dir, f).path)
                    val out = p.analyse(bmp)
                    tops += out.field.topCellScore; totals += out.timings.totalMs
                    crop += out.timings.cropMs; classify += out.timings.classifyMs; detect += out.timings.detectMs
                    bmp.recycle()
                }
                if (reference == null) reference = tops
                val maxDiff = tops.indices.maxOf { kotlin.math.abs(tops[it] - reference!![it]) }
                val sorted = totals.sorted()
                Log.i(
                    "WideFieldBench",
                    "$name | ${files.size} fields: median ${sorted[sorted.size / 2]} ms, mean ${totals.average().toLong()} ms, max ${sorted.last()} ms | " +
                        "mean detect ${detect / files.size}, crop ${crop / files.size}, classify ${classify / files.size} ms | max top-score diff vs reference ${"%.6f".format(maxDiff)}",
                )
            } finally {
                p.close()
            }
        }
    }
}
