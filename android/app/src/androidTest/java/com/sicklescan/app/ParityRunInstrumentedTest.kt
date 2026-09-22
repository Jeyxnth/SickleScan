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
 * Device-side half of the classifier preprocessing verification (Phase 14d). Reads files/parity/jobs.csv ("relative/path,model" per line, model =
 * sickle | malaria | guardrail), decodes each image with BitmapFactory exactly as the app does, classifies it with the app's real ImageClassifier
 * (the new TensorFlow-exact resize), and writes files/parity/results.csv ("path,model,P,classifyMs"). scripts/device/analyze_parity_results.py
 * compares it with Python's TensorFlow preprocessing. Skipped if no job list was pushed. Progress: `adb logcat -d -s ParityRun`.
 */
@RunWith(AndroidJUnit4::class)
class ParityRunInstrumentedTest {

    @Test
    fun runJobs() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = File(ctx.getExternalFilesDir(null), "parity")
        val jobs = File(dir, "jobs.csv")
        assumeTrue("no jobs pushed to ${jobs.path}", jobs.exists())
        val classifiers = mutableMapOf<String, ImageClassifier>()
        fun classifier(model: String) = classifiers.getOrPut(model) {
            when (model) {
                "sickle" -> ImageClassifier(ctx, Disease.SICKLE_CELL)
                "malaria" -> ImageClassifier(ctx, Disease.MALARIA)
                else -> ImageClassifier(ctx, "guardrail_model.tflite", "guardrail_labels.txt")
            }
        }
        val out = File(dir, "results.csv").bufferedWriter()
        try {
            out.write("path,model,P,classifyMs\n")
            var n = 0
            for (line in jobs.readLines().filter { it.isNotBlank() }) {
                val (rel, model) = line.split(",")
                val bmp = BitmapFactory.decodeFile(File(dir, rel).path)
                if (bmp == null) {
                    out.write("$rel,$model,NaN,0\n")
                    continue
                }
                val t0 = System.nanoTime()
                val p = classifier(model).classify(bmp)
                out.write("$rel,$model,$p,${(System.nanoTime() - t0) / 1_000_000}\n")
                bmp.recycle()
                if (++n % 200 == 0) {
                    out.flush()
                    Log.i(TAG, "$n jobs done")
                }
            }
            Log.i(TAG, "finished $n jobs")
        } finally {
            out.close()
            classifiers.values.forEach { it.close() }
        }
    }

    companion object {
        private const val TAG = "ParityRun"
    }
}
