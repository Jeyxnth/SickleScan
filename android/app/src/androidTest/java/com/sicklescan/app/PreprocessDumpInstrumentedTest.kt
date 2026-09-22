package com.sicklescan.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Diagnostic (not a pass/fail test): dumps what the phone actually feeds the models, to compare with the Python preprocessing.
 * For every image in the test APK's asset folders it writes, under files/preprocess_dump/:
 *   <name>.decoded  = width, height (ints) then ARGB ints of the BitmapFactory-decoded bitmap
 *   <name>.tensor   = the 224x224x3 float32 tensor produced by the SAME processor ImageClassifier builds
 *                     (TensorImage(FLOAT32) -> ResizeOp(224,224,BILINEAR) -> NormalizeOp(127.5,127.5))
 * and, for every <name>.pytensor found next to them (the Python-preprocessed tensor pushed from the PC), runs the sickle model ON THE
 * PHONE on it and logs the probability. Read with `adb logcat -d -s PreprocessDump`.
 */
@RunWith(AndroidJUnit4::class)
class PreprocessDumpInstrumentedTest {

    private val folders = listOf("sample_test_images", "malaria_test_images", "guardrail_test_images")

    private fun floats(buf: ByteBuffer, n: Int): FloatArray {
        val out = FloatArray(n)
        buf.order(ByteOrder.nativeOrder()).asFloatBuffer().get(out)
        return out
    }

    @Test
    fun dumpPreprocessing() {
        val inst = InstrumentationRegistry.getInstrumentation()
        val ctx = inst.targetContext
        val outDir = File(ctx.getExternalFilesDir(null), "preprocess_dump").apply { mkdirs() }
        // identical to ImageClassifier's processor
        val processor = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(127.5f, 127.5f))
            .build()
        val sickle = Interpreter(FileUtil.loadMappedFile(ctx, Disease.SICKLE_CELL.modelAsset))
        try {
            for (folder in folders) {
                for (name in inst.context.assets.list(folder).orEmpty().sorted()) {
                    val bmp = inst.context.assets.open("$folder/$name").use { BitmapFactory.decodeStream(it) } ?: continue
                    val tag = "${folder}__${name}"
                    // 1) decoded pixels
                    val px = IntArray(bmp.width * bmp.height)
                    bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
                    val bb = ByteBuffer.allocate(8 + px.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                    bb.putInt(bmp.width).putInt(bmp.height); px.forEach { bb.putInt(it) }
                    File(outDir, "$tag.decoded").writeBytes(bb.array())
                    // 2) the tensor the app feeds the model
                    var ti = TensorImage(DataType.FLOAT32)
                    ti.load(bmp)
                    ti = processor.process(ti)
                    val t = ti.tensorBuffer.floatArray
                    val tb = ByteBuffer.allocate(t.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                    t.forEach { tb.putFloat(it) }
                    File(outDir, "$tag.tensor").writeBytes(tb.array())
                    val out = Array(1) { FloatArray(1) }
                    ti.buffer.rewind(); sickle.run(ti.buffer, out)
                    Log.i(TAG, "$tag ${bmp.width}x${bmp.height} config=${bmp.config} alpha=${bmp.hasAlpha()} cs=${if (android.os.Build.VERSION.SDK_INT >= 26) bmp.colorSpace?.name else "n/a"} app-preprocessing sickle P=${"%.6f".format(out[0][0])}")
                    // 3) the Python-preprocessed tensor for the same image (pushed by the PC), through the same on-device model
                    val py = File(outDir, "$tag.pytensor")
                    if (py.exists()) {
                        val pb = ByteBuffer.allocateDirect(224 * 224 * 3 * 4).order(ByteOrder.nativeOrder())
                        pb.put(py.readBytes()); pb.rewind()
                        val o2 = Array(1) { FloatArray(1) }
                        sickle.run(pb, o2)
                        Log.i(TAG, "$tag python-preprocessing (on phone) sickle P=${"%.6f".format(o2[0][0])}")
                    }
                    bmp.recycle()
                }
            }
        } finally {
            sickle.close()
        }
    }

    companion object {
        private const val TAG = "PreprocessDump"
    }
}
