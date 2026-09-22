package com.sicklescan.app

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The bundled CenterNet-style cell locator (Phase 11a, exported in Phase 13): a MobileNetV2 + light FPN detector with a
 * fixed 640x640 input (see [ImageOps.letterbox]) and a [1, 160, 160, 5] output (cell probability, box size, centre offset;
 * the sigmoid is inside the model). Float16 weights. Peak picking is in [WideFieldMalaria.decodeDetections].
 */
class CellDetector(context: Context, numThreads: Int) : AutoCloseable {

    class DetectorException(message: String, cause: Throwable? = null) : Exception(message, cause)

    private val interpreter: Interpreter
    private val inputBuffer: ByteBuffer
    private val outputBuffer: ByteBuffer
    private val gridH: Int
    private val gridW: Int

    init {
        try {
            interpreter = Interpreter(
                FileUtil.loadMappedFile(context, ASSET),
                Interpreter.Options().setNumThreads(numThreads),
            )
            val inShape = interpreter.getInputTensor(0).shape()   // [1, 640, 640, 3]
            val outShape = interpreter.getOutputTensor(0).shape() // [1, 160, 160, 5]
            if (inShape.size != 4 || inShape[1] != WideFieldMalaria.DETECTOR_INPUT_SIZE ||
                inShape[2] != WideFieldMalaria.DETECTOR_INPUT_SIZE || inShape[3] != 3 || outShape.size != 4 || outShape[3] != 5
            ) {
                throw DetectorException("Unexpected detector tensor shapes: in=${inShape.toList()} out=${outShape.toList()}")
            }
            gridH = outShape[1]
            gridW = outShape[2]
            inputBuffer = ByteBuffer.allocateDirect(inShape.fold(4) { a, b -> a * b }).order(ByteOrder.nativeOrder())
            outputBuffer = ByteBuffer.allocateDirect(outShape.fold(4) { a, b -> a * b }).order(ByteOrder.nativeOrder())
        } catch (e: DetectorException) {
            throw e
        } catch (e: Exception) {
            throw DetectorException("Failed to load the on-device cell detector", e)
        }
    }

    /** Runs the detector on a letterboxed canvas; returns detections in canvas pixels (strongest first). */
    fun detect(canvas: ImageOps.Letterboxed): List<WideFieldMalaria.Detection> {
        try {
            inputBuffer.rewind()
            inputBuffer.asFloatBuffer().put(canvas.data)
            outputBuffer.rewind()
            interpreter.run(inputBuffer, outputBuffer)
            outputBuffer.rewind()
            val map = FloatArray(gridH * gridW * 5)
            outputBuffer.asFloatBuffer().get(map)
            return WideFieldMalaria.decodeDetections(map, gridH, gridW)
        } catch (e: Exception) {
            throw DetectorException("Cell detection failed", e)
        }
    }

    override fun close() {
        interpreter.close()
    }

    companion object {
        const val ASSET = "cell_detector.tflite"
    }
}
