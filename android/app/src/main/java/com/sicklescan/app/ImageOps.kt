package com.sicklescan.app

import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Pixel resampling for the wide-field malaria pipeline. Pure Kotlin on ARGB int arrays (no Android
 * dependency) so it is a real local unit test (ImageOpsTest) against golden values produced by the
 * OpenCV code the Python validation used (Phases 11a-c). Matching OpenCV matters: the classifier was
 * validated on crops resized with cv2.resize -- INTER_AREA when shrinking (each output pixel is the
 * average of the source area it covers, with fractional edge weights), INTER_LINEAR when enlarging
 * (bilinear on half-pixel centres, borders replicated) -- and the reported false-alarm / sensitivity
 * numbers only apply if the deployed pixels are (nearly) the same. Results are rounded to whole 0-255
 * values like cv2's uint8 output, then scaled to [-1, 1] with (v - 127.5) / 127.5.
 */
object ImageOps {

    /** Detector canvas: the photo's longer side is scaled to [size], pasted top-left on a white square. */
    class Letterboxed(val data: FloatArray, val size: Int, val scale: Float)

    fun letterbox(argb: IntArray, width: Int, height: Int, size: Int): Letterboxed {
        val scale = size.toFloat() / max(width, height)
        val newW = Math.rint(width * scale.toDouble()).toInt().coerceIn(1, size)
        val newH = Math.rint(height * scale.toDouble()).toInt().coerceIn(1, size)
        val out = FloatArray(size * size * 3) { 1f } // white canvas: (255 - 127.5) / 127.5
        val rgb = resize(argb, width, height, 0, 0, width, height, newW, newH)
        for (y in 0 until newH) {
            for (x in 0 until newW) {
                val o = (y * size + x) * 3
                val i = (y * newW + x) * 3
                out[o] = norm(rgb[i]); out[o + 1] = norm(rgb[i + 1]); out[o + 2] = norm(rgb[i + 2])
            }
        }
        return Letterboxed(out, size, scale)
    }

    /**
     * The classifier input for one detected cell, matching the Python crop used in validation: a square window
     * centred on the box with side = box side x [padding] (at least 16 px), border pixels replicated outside
     * the image, resized to [outSize] x [outSize] (INTER_AREA when the window is larger than outSize, else
     * INTER_LINEAR), RGB, scaled to [-1, 1]. Written into [out] (NHWC) starting at [outOffset]; [out] must have room for
     * outSize*outSize*3 floats from there, so several crops can fill one batch buffer (from different threads: this is pure).
     */
    fun cropForClassifier(
        argb: IntArray, width: Int, height: Int,
        boxX0: Float, boxY0: Float, boxX1: Float, boxY1: Float,
        out: FloatArray, outSize: Int = 224, padding: Double = 1.15, outOffset: Int = 0,
    ) {
        val side = max(16, Math.rint(max(boxY1 - boxY0, boxX1 - boxX0) * padding).toInt())
        val cy = (boxY0 + boxY1) / 2.0
        val cx = (boxX0 + boxX1) / 2.0
        val py0 = Math.rint(cy - side / 2.0).toInt()
        val px0 = Math.rint(cx - side / 2.0).toInt()
        val rgb = resize(argb, width, height, px0, py0, side, side, outSize, outSize)
        for (i in 0 until outSize * outSize * 3) out[outOffset + i] = norm(rgb[i])
    }

    /**
     * TensorFlow's `tf.image.resize(img, [h, w])` with its default bilinear method, which is what the sickle-cell, NIH-malaria and
     * guardrail models were trained and validated with (scripts/data_pipeline.py): output pixel o samples the source at
     * (o + 0.5) * scale - 0.5 (half-pixel centres) with two taps, edge pixels clamped, **two taps even when shrinking (no area
     * averaging or antialiasing)**, and the result stays a float (no rounding to whole 0-255 values). This is deliberately NOT the
     * OpenCV-style [resize] above (area averaging when shrinking, whole-number output), which matches the wide-field crops' own pipeline.
     * Android's Bitmap.createScaledBitmap, which TFLite-Support's ResizeOp used before, is an 8-bit fixed-point bilinear (weights in
     * 1/16 steps, truncated) and differs from this by about 0.5 gray level on average (Phase 14c).
     *
     * The source is read one row at a time through [readRow] (row index, destination IntArray of ARGB pixels), so a 12-megapixel photo
     * never needs a second full-size pixel array. [out] receives dstH*dstW*3 floats, RGB, values in 0..255.
     */
    fun resizeBilinearTf(
        readRow: (y: Int, dest: IntArray) -> Unit, srcW: Int, srcH: Int, dstW: Int, dstH: Int, out: FloatArray,
    ) {
        val xw = axisWeights(srcW, dstW, false)
        val yw = axisWeights(srcH, dstH, false)
        val rowBuf = IntArray(srcW)
        val hRows = HashMap<Int, FloatArray>()
        fun hResampled(y: Int): FloatArray = hRows.getOrPut(y) {
            readRow(y, rowBuf)
            val r = FloatArray(dstW * 3)
            for (dx in 0 until dstW) {
                val t = xw[dx]
                var sr = 0f; var sg = 0f; var sb = 0f
                for (k in t.weights.indices) {
                    val p = rowBuf[t.start + k]
                    val wt = t.weights[k]
                    sr += wt * ((p shr 16) and 0xFF); sg += wt * ((p shr 8) and 0xFF); sb += wt * (p and 0xFF)
                }
                r[dx * 3] = sr; r[dx * 3 + 1] = sg; r[dx * 3 + 2] = sb
            }
            r
        }
        for (dy in 0 until dstH) {
            val t = yw[dy]
            val rows = Array(t.weights.size) { hResampled(t.start + it) }
            for (i in 0 until dstW * 3) {
                var v = 0f
                for (k in rows.indices) v += t.weights[k] * rows[k][i]
                out[dy * dstW * 3 + i] = v
            }
            // drop rows no later output row needs (keeps the cache at a few rows when shrinking)
            if (dy + 1 < dstH) {
                val nextStart = yw[dy + 1].start
                hRows.keys.removeAll { it < nextStart }
            }
        }
    }

    /**
     * The complete classifier preprocessing (sickle cell, malaria single-cell, guardrail): [resizeBilinearTf], then `(v - 127.5) / 127.5`
     * (Keras `mobilenet_v2.preprocess_input`, mode "tf"), in place in [out] (dstH*dstW*3 floats in [-1, 1]). The app and the JVM parity
     * test both call exactly this.
     */
    fun classifierInput(readRow: (y: Int, dest: IntArray) -> Unit, srcW: Int, srcH: Int, dstW: Int, dstH: Int, out: FloatArray) {
        resizeBilinearTf(readRow, srcW, srcH, dstW, dstH, out)
        for (i in out.indices) out[i] = (out[i] - 127.5f) / 127.5f
    }

    /** Convenience for tests / in-memory pixels: [resizeBilinearTf] over an ARGB array. */
    fun resizeBilinearTf(argb: IntArray, width: Int, height: Int, dstW: Int, dstH: Int): FloatArray {
        val out = FloatArray(dstW * dstH * 3)
        resizeBilinearTf({ y, dest -> System.arraycopy(argb, y * width, dest, 0, width) }, width, height, dstW, dstH, out)
        return out
    }

    private fun norm(v: Int): Float = (v - 127.5f) / 127.5f

    /**
     * Resizes the [srcW] x [srcH] window whose top-left is (x0, y0) in the image to [dstW] x [dstH]; window pixels
     * outside the image take the nearest edge pixel (replicated border). Returns RGB bytes as ints, row-major, 3 per pixel.
     */
    private fun resize(
        argb: IntArray, imgW: Int, imgH: Int, x0: Int, y0: Int, srcW: Int, srcH: Int, dstW: Int, dstH: Int,
    ): IntArray {
        val shrinking = srcW > dstW || srcH > dstH
        val xw = axisWeights(srcW, dstW, shrinking)
        val yw = axisWeights(srcH, dstH, shrinking)
        // Horizontal pass over only the source rows that are needed, then vertical pass.
        val rowLo = yw.minOf { it.start }
        val rowHi = yw.maxOf { it.start + it.weights.size - 1 }
        val rows = rowHi - rowLo + 1
        val tmp = FloatArray(rows * dstW * 3)
        for (r in 0 until rows) {
            val iy = (y0 + rowLo + r).coerceIn(0, imgH - 1)
            val base = iy * imgW
            for (dx in 0 until dstW) {
                val w = xw[dx]
                var sr = 0f; var sg = 0f; var sb = 0f
                for (k in w.weights.indices) {
                    val ix = (x0 + w.start + k).coerceIn(0, imgW - 1)
                    val p = argb[base + ix]
                    val wt = w.weights[k]
                    sr += wt * ((p shr 16) and 0xFF); sg += wt * ((p shr 8) and 0xFF); sb += wt * (p and 0xFF)
                }
                val o = (r * dstW + dx) * 3
                tmp[o] = sr; tmp[o + 1] = sg; tmp[o + 2] = sb
            }
        }
        val out = IntArray(dstW * dstH * 3)
        for (dy in 0 until dstH) {
            val w = yw[dy]
            for (dx in 0 until dstW) {
                var sr = 0f; var sg = 0f; var sb = 0f
                for (k in w.weights.indices) {
                    val o = ((w.start - rowLo + k) * dstW + dx) * 3
                    val wt = w.weights[k]
                    sr += wt * tmp[o]; sg += wt * tmp[o + 1]; sb += wt * tmp[o + 2]
                }
                val o = (dy * dstW + dx) * 3
                out[o] = Math.rint(sr.toDouble()).toInt().coerceIn(0, 255)
                out[o + 1] = Math.rint(sg.toDouble()).toInt().coerceIn(0, 255)
                out[o + 2] = Math.rint(sb.toDouble()).toInt().coerceIn(0, 255)
            }
        }
        return out
    }

    /** Source taps for one output pixel: window coordinates [start, start + weights.size), weights sum to 1. */
    private class Taps(val start: Int, val weights: FloatArray)

    private fun axisWeights(srcLen: Int, dstLen: Int, area: Boolean): Array<Taps> {
        val scale = srcLen.toDouble() / dstLen
        return Array(dstLen) { i ->
            if (area) {
                // INTER_AREA: overlap of [i*scale, (i+1)*scale) with each source pixel, normalised.
                val s = i * scale
                val e = min((i + 1) * scale, srcLen.toDouble())
                val first = floor(s).toInt()
                val last = min(ceil(e).toInt() - 1, srcLen - 1)
                val w = FloatArray(last - first + 1) { k ->
                    (min(first + k + 1.0, e) - max(first + k.toDouble(), s)).toFloat()
                }
                val total = w.sum()
                Taps(first, FloatArray(w.size) { w[it] / total })
            } else {
                // INTER_LINEAR: half-pixel centres, replicated border (indices clamped to the window).
                val src = (i + 0.5) * scale - 0.5
                val f = floor(src)
                val frac = (src - f).toFloat()
                val i0 = f.toInt().coerceIn(0, srcLen - 1)
                val i1 = (f.toInt() + 1).coerceIn(0, srcLen - 1)
                if (i0 == i1) Taps(i0, floatArrayOf(1f)) else Taps(i0, floatArrayOf(1f - frac, frac))
            }
        }
    }
}
