package com.sicklescan.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * Parity of the Kotlin resampling with the OpenCV code the Phase 11 validation used. Golden values come from
 * scripts/detector/make_kotlin_goldens.py (export_tflite.letterbox and build_crops.crop run on a deterministic
 * synthetic image regenerated here by the same integer formula). Tolerance: OpenCV's bilinear uses fixed-point
 * weights, so a difference of a couple of 0-255 levels is expected; the mean difference must stay tiny.
 */
class ImageOpsTest {

    private fun synth(w: Int, h: Int): IntArray = IntArray(w * h) { i ->
        val x = i % w
        val y = i / w
        fun v(c: Int) = (x * 7 + y * 13 + ((x * y) % 37) * 5 + c * 50) % 256
        (0xFF shl 24) or (v(0) shl 16) or (v(1) shl 8) or v(2)
    }

    private fun goldenLines(name: String): List<String> =
        checkNotNull(javaClass.getResourceAsStream("/goldens/$name")) { "missing golden $name" }
            .bufferedReader().readLines().filter { it.isNotBlank() }

    private fun toUint8(f: Float) = Math.rint(f * 127.5 + 127.5).toInt()

    @Test
    fun `letterbox matches OpenCV for shrinking, enlarging and square inputs`() {
        val lines = goldenLines("letterbox.txt")
        assertEquals(8, lines.size)
        for (k in 0 until 4) {
            val head = lines[2 * k].split(" ")
            val w = head[1].toInt()
            val h = head[2].toInt()
            val size = head[3].toInt()
            val scale = head[4].removePrefix("scale=").toFloat()
            val expected = lines[2 * k + 1].split(" ").map { it.toInt() }
            val out = ImageOps.letterbox(synth(w, h), w, h, size)
            assertEquals("scale for ${w}x$h -> $size", scale, out.scale, 1e-6f)
            assertEquals(expected.size, out.data.size)
            var maxDiff = 0
            var sum = 0L
            for (i in expected.indices) {
                val d = abs(toUint8(out.data[i]) - expected[i])
                maxDiff = maxOf(maxDiff, d)
                sum += d
            }
            assertTrue("letterbox ${w}x$h -> $size: max diff $maxDiff", maxDiff <= 2)
            assertTrue("letterbox ${w}x$h -> $size: mean diff ${sum.toDouble() / expected.size}", sum.toDouble() / expected.size < 0.5)
        }
    }

    @Test
    fun `classifier crops match OpenCV including edge replication, tiny boxes and the shrink path`() {
        val lines = goldenLines("crops.txt")
        val (w, h) = lines[0].split(" ").drop(1).map { it.toInt() }
        val pixels = synth(w, h)
        val out = FloatArray(224 * 224 * 3)
        var n = 0
        for (k in 1 until lines.size step 2) {
            val head = lines[k].split(" ")
            val name = head[0]
            val (x0, y0, x1, y1) = head.subList(1, 5).map { it.toFloat() }
            val expected = lines[k + 1].split(" ").map { it.toInt() } // 32 x 32 x 3 subsample, every 7th pixel
            ImageOps.cropForClassifier(pixels, w, h, x0, y0, x1, y1, out)
            var maxDiff = 0
            var sum = 0L
            var idx = 0
            for (yy in 0 until 224 step 7) for (xx in 0 until 224 step 7) for (c in 0..2) {
                val d = abs(toUint8(out[(yy * 224 + xx) * 3 + c]) - expected[idx++])
                maxDiff = maxOf(maxDiff, d)
                sum += d
            }
            assertTrue("crop '$name': max diff $maxDiff", maxDiff <= 2)
            assertTrue("crop '$name': mean diff ${sum.toDouble() / expected.size}", sum.toDouble() / expected.size < 0.5)
            n++
        }
        assertEquals("all golden cases exercised", 6, n)
    }
}

/**
 * The classifiers' resize must equal TensorFlow's `tf.image.resize` (default bilinear), the call they were trained and validated with.
 * Goldens are the real tf.image.resize output (scripts/detector/make_tf_goldens.py) on the same deterministic synthetic image.
 * Unlike the OpenCV-style path, this keeps floats and uses two taps even when shrinking, so the tolerance is float rounding only.
 */
class TfBilinearResizeTest {

    private fun synth(w: Int, h: Int): IntArray = IntArray(w * h) { i ->
        val x = i % w
        val y = i / w
        fun v(c: Int) = (x * 7 + y * 13 + ((x * y) % 37) * 5 + c * 50) % 256
        (0xFF shl 24) or (v(0) shl 16) or (v(1) shl 8) or v(2)
    }

    @Test
    fun `matches tf image resize for upscaling, non-square, mild and strong shrinking`() {
        val lines = checkNotNull(javaClass.getResourceAsStream("/goldens/tf_resize.txt")).bufferedReader().readLines().filter { it.isNotBlank() }
        assertEquals(12, lines.size)
        for (k in 0 until 6) {
            val (w, h) = lines[2 * k].split(" ").drop(1).map { it.toInt() }
            val expected = lines[2 * k + 1].split(" ").map { it.toFloat() } // 32 x 32 x 3 subsample, every 7th pixel
            val out = ImageOps.resizeBilinearTf(synth(w, h), w, h, 224, 224)
            var maxDiff = 0f
            var idx = 0
            for (y in 0 until 224 step 7) for (x in 0 until 224 step 7) for (c in 0..2) {
                maxDiff = maxOf(maxDiff, abs(out[(y * 224 + x) * 3 + c] - expected[idx++]))
            }
            // 0.05 gray level = 0.02% of the range: TensorFlow computes the sample position in float32, this code in double, and the synthetic
            // image is pure noise (neighbouring pixels differ by up to 255), so a ~1e-5 pixel position difference shows up as ~0.01-0.02.
            assertTrue("tf.image.resize parity ${w}x$h -> 224: max diff $maxDiff gray levels", maxDiff < 0.05f)
        }
    }

    @Test
    fun `output stays a float, not rounded to whole gray levels`() {
        val out = ImageOps.resizeBilinearTf(synth(53, 41), 53, 41, 224, 224)
        assertTrue("some values must be fractional", out.any { it != Math.rint(it.toDouble()).toFloat() })
        assertTrue(out.all { it in 0f..255f })
    }

    @Test
    fun `row-by-row reading gives the same result as the in-memory version`() {
        val w = 300; val h = 260
        val px = synth(w, h)
        val a = ImageOps.resizeBilinearTf(px, w, h, 224, 224)
        val b = FloatArray(224 * 224 * 3)
        ImageOps.resizeBilinearTf({ y, dest -> System.arraycopy(px, y * w, dest, 0, w) }, w, h, 224, 224, b)
        assertTrue(a.contentEquals(b))
    }
}
