package com.sicklescan.app

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

/**
 * Wide-field malaria, fully on device (no network): photo -> letterbox 640 -> [CellDetector] -> one 224x224 crop per
 * detected cell (taken from the full working-resolution photo, not the 640 canvas) -> the bundled BBBC041 malaria
 * classifier (the same malaria_model.tflite the single-cell path uses) -> [WideFieldMalaria.interpret].
 * Crops are built in parallel and classified in small batches on a background thread. None of these speed settings
 * changes a score (measured on a phone: identical top scores to the one-at-a-time version). Per-stage timings are recorded
 * so the real on-device latency can be read straight off the result (Phase 13 Task 4). Measured on a Snapdragon 7 Gen 3
 * phone: about 2.9 s per BBBC-size field (about 75 cells), of which the classifier is about 1.95 s.
 */
class WideFieldMalariaPipeline(context: Context, private val config: Config = Config()) : AutoCloseable {

    /** Speed knobs only: none of them changes any score. [batchSize] crops go through the classifier per call; [cropThreads] build the crops in parallel. */
    data class Config(val threads: Int = DEFAULT_THREADS, val batchSize: Int = DEFAULT_BATCH, val cropThreads: Int = DEFAULT_CROP_THREADS)

    class PipelineException(message: String, cause: Throwable? = null) : Exception(message, cause)

    enum class Stage { PREPARING, FINDING_CELLS, CHECKING_CELLS }

    class Timings(val prepareMs: Long, val detectMs: Long, val cropMs: Long, val classifyMs: Long, val totalMs: Long)

    class Output(val field: WideFieldMalaria.FieldResult, val cellScores: FloatArray, val timings: Timings)

    private val batch = config.batchSize.coerceAtLeast(1)
    private val detector = CellDetector(context, config.threads)
    private val classifier: Interpreter
    private val batchInput = FloatArray(batch * CROP * CROP * 3)
    private val batchBuffer: ByteBuffer = ByteBuffer.allocateDirect(batch * CROP * CROP * 3 * 4).order(ByteOrder.nativeOrder())
    private val batchScores = Array(batch) { FloatArray(1) }
    private val cropPool = if (config.cropThreads > 1) Executors.newFixedThreadPool(config.cropThreads) else null

    init {
        try {
            classifier = Interpreter(
                FileUtil.loadMappedFile(context, Disease.MALARIA.modelAsset),
                Interpreter.Options().setNumThreads(config.threads),
            )
            val shape = classifier.getInputTensor(0).shape()
            if (shape.size != 4 || shape[1] != CROP || shape[2] != CROP || shape[3] != 3) {
                throw PipelineException("Unexpected malaria classifier input shape ${shape.toList()}")
            }
            if (batch > 1) {
                classifier.resizeInput(0, intArrayOf(batch, CROP, CROP, 3))
                classifier.allocateTensors()
            }
        } catch (e: PipelineException) {
            detector.close()
            throw e
        } catch (e: Exception) {
            detector.close()
            throw PipelineException("Failed to load the malaria classifier", e)
        }
    }

    /**
     * Pays the one-time first-use cost (kernel setup, first inference) before the user taps Analyze: the first analysis on a
     * fresh pipeline took 7.3 s vs about 3 s afterwards. Runs the detector on a blank canvas and the crop + classifier path on a
     * synthetic image; results are discarded. Safe to call from a background thread.
     */
    @Synchronized
    fun warmUp() {
        detector.detect(ImageOps.Letterboxed(FloatArray(WideFieldMalaria.DETECTOR_INPUT_SIZE * WideFieldMalaria.DETECTOR_INPUT_SIZE * 3) { 1f },
            WideFieldMalaria.DETECTOR_INPUT_SIZE, 1f))
        // Exercise the crop code too (parallel, on a synthetic image): it is plain Kotlin that runs slowly until Android's JIT
        // has compiled it, which otherwise lands inside the user's first analysis.
        val w = 800
        val h = 600
        val fake = IntArray(w * h) { (0xFF shl 24) or ((it * 7) and 0xFFFFFF) }
        repeat(2) {
            val jobs = (0 until batch).map { j ->
                Callable { ImageOps.cropForClassifier(fake, w, h, 100f + 40 * j, 100f + 30 * j, 220f + 40 * j, 210f + 30 * j, batchInput, outOffset = j * CROP * CROP * 3) }
            }
            if (cropPool != null && batch > 1) cropPool.invokeAll(jobs).forEach { it.get() } else jobs.forEach { it.call() }
            batchBuffer.rewind()
            batchBuffer.asFloatBuffer().put(batchInput)
            batchBuffer.rewind()
            classifier.run(batchBuffer, batchScores)
        }
    }

    /** Blocking; call from a background thread. [onProgress] is invoked from that thread (stage, done, total). */
    @Synchronized
    fun analyse(bitmap: Bitmap, onProgress: (Stage, Int, Int) -> Unit = { _, _, _ -> }): Output {
        try {
            val t0 = System.nanoTime()
            onProgress(Stage.PREPARING, 0, 0)
            val work = workingCopy(bitmap)
            val w = work.width
            val h = work.height
            val pixels = IntArray(w * h)
            work.getPixels(pixels, 0, w, 0, 0, w, h)
            if (work !== bitmap) work.recycle()
            val canvas = ImageOps.letterbox(pixels, w, h, WideFieldMalaria.DETECTOR_INPUT_SIZE)
            val t1 = System.nanoTime()

            onProgress(Stage.FINDING_CELLS, 0, 0)
            val detections = detector.detect(canvas)
            val t2 = System.nanoTime()

            val scores = FloatArray(detections.size)
            var cropNs = 0L
            var classifyNs = 0L
            val inv = 1f / canvas.scale
            val stride = CROP * CROP * 3
            var start = 0
            while (start < detections.size) {
                onProgress(Stage.CHECKING_CELLS, start, detections.size)
                val count = min(batch, detections.size - start)
                val c0 = System.nanoTime()
                val jobs = (0 until count).map { j ->
                    Callable {
                        val d = detections[start + j]
                        ImageOps.cropForClassifier(pixels, w, h, d.x0 * inv, d.y0 * inv, d.x1 * inv, d.y1 * inv, batchInput, outOffset = j * stride)
                    }
                }
                if (cropPool != null && count > 1) cropPool.invokeAll(jobs).forEach { it.get() } else jobs.forEach { it.call() }
                batchBuffer.rewind()
                batchBuffer.asFloatBuffer().put(batchInput)
                val c1 = System.nanoTime()
                batchBuffer.rewind()
                classifier.run(batchBuffer, batchScores) // slots beyond [count] (last, partial batch) hold stale crops; ignored
                for (j in 0 until count) scores[start + j] = batchScores[j][0]
                val c2 = System.nanoTime()
                cropNs += c1 - c0
                classifyNs += c2 - c1
                start += count
            }
            if (detections.isNotEmpty()) onProgress(Stage.CHECKING_CELLS, detections.size, detections.size)
            val t3 = System.nanoTime()

            return Output(
                field = WideFieldMalaria.interpret(scores),
                cellScores = scores,
                timings = Timings(
                    prepareMs = (t1 - t0) / 1_000_000, detectMs = (t2 - t1) / 1_000_000,
                    cropMs = cropNs / 1_000_000, classifyMs = classifyNs / 1_000_000, totalMs = (t3 - t0) / 1_000_000,
                ),
            )
        } catch (e: CellDetector.DetectorException) {
            throw PipelineException("Cell detection failed", e)
        } catch (e: PipelineException) {
            throw e
        } catch (e: Exception) {
            throw PipelineException("Wide-field analysis failed", e)
        }
    }

    /** The photo itself if it is within the working-resolution cap, else a bilinear downscale of it (bounds memory). */
    private fun workingCopy(bitmap: Bitmap): Bitmap {
        val longSide = max(bitmap.width, bitmap.height)
        if (longSide <= WideFieldMalaria.WORKING_LONG_SIDE) return bitmap
        val s = WideFieldMalaria.WORKING_LONG_SIDE.toFloat() / longSide
        return Bitmap.createScaledBitmap(bitmap, Math.round(bitmap.width * s), Math.round(bitmap.height * s), true)
    }

    override fun close() {
        cropPool?.shutdown()
        detector.close()
        classifier.close()
    }

    companion object {
        private const val CROP = 224
        const val DEFAULT_THREADS = 4
        const val DEFAULT_BATCH = 8
        val DEFAULT_CROP_THREADS = Runtime.getRuntime().availableProcessors().coerceIn(2, 6)
    }
}
