package com.sicklescan.app

import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Verification harness, not a regular test (skipped unless PARITY_IN / PARITY_OUT are set): runs the EXACT classifier preprocessing the app
 * uses ([ImageOps.classifierInput]) on the JVM over raw RGB dumps written by scripts/device/verify_kotlin_resize_jvm.py (int32 width, int32
 * height, then w*h*3 bytes RGB, i.e. the pixels TensorFlow decoded), and writes each resulting 224x224x3 tensor as little-endian float32,
 * so Python can push it through the real models and compare with TensorFlow's own preprocessing. It isolates the resize algorithm from any
 * decoder difference.
 */
class ClassifierPreprocessJvmHarnessTest {

    @Test
    fun writeKotlinTensors() {
        val inDir = System.getenv("PARITY_IN")
        val outDir = System.getenv("PARITY_OUT")
        assumeTrue("PARITY_IN / PARITY_OUT not set", inDir != null && outDir != null)
        File(outDir!!).mkdirs()
        for (f in File(inDir!!).listFiles { x -> x.extension == "raw" }!!.sortedBy { it.name }) {
            val b = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
            val w = b.int
            val h = b.int
            val rgb = ByteArray(w * h * 3).also { b.get(it) }
            val out = FloatArray(224 * 224 * 3)
            ImageOps.classifierInput({ y, dest ->
                for (x in 0 until w) {
                    val i = (y * w + x) * 3
                    dest[x] = (0xFF shl 24) or ((rgb[i].toInt() and 0xFF) shl 16) or ((rgb[i + 1].toInt() and 0xFF) shl 8) or (rgb[i + 2].toInt() and 0xFF)
                }
            }, w, h, 224, 224, out)
            val ob = ByteBuffer.allocate(out.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            out.forEach { ob.putFloat(it) }
            File(outDir, f.nameWithoutExtension + ".f32").writeBytes(ob.array())
        }
    }
}
