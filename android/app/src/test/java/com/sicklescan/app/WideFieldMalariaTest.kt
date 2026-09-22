package com.sicklescan.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class WideFieldMalariaTest {

    /** Builds a detector output map [h, w, 5] with the given peaks: (x, y, prob, w, h, dx, dy). */
    private fun map(h: Int, w: Int, vararg peaks: FloatArray): FloatArray {
        val m = FloatArray(h * w * 5)
        for (p in peaks) {
            val o = (p[1].toInt() * w + p[0].toInt()) * 5
            m[o] = p[2]; m[o + 1] = p[3]; m[o + 2] = p[4]; m[o + 3] = p[5]; m[o + 4] = p[6]
        }
        return m
    }

    @Test
    fun `operating point is the validated one`() {
        assertEquals(0.985f, WideFieldMalaria.FIELD_THRESHOLD, 0f)
        assertEquals(0.3f, WideFieldMalaria.DETECTOR_SCORE_THRESHOLD, 0f)
        assertEquals(640, WideFieldMalaria.DETECTOR_INPUT_SIZE)
    }

    @Test
    fun `decode turns a peak into a box in canvas pixels`() {
        // grid cell (10, 5), prob 0.9, 10 x 12 grid cells wide/high, centre offset (0.5, 0.25)
        val d = WideFieldMalaria.decodeDetections(map(20, 20, floatArrayOf(10f, 5f, 0.9f, 10f, 12f, 0.5f, 0.25f)), 20, 20)
        assertEquals(1, d.size)
        val b = d[0]
        val cx = (10 + 0.5f) * 4
        val cy = (5 + 0.25f) * 4
        assertEquals(cx - 20f, b.x0, 1e-4f); assertEquals(cx + 20f, b.x1, 1e-4f)
        assertEquals(cy - 24f, b.y0, 1e-4f); assertEquals(cy + 24f, b.y1, 1e-4f)
        assertEquals(0.9f, b.score, 0f)
    }

    @Test
    fun `decode keeps only local maxima above the score threshold, strongest first`() {
        val m = map(
            20, 20,
            floatArrayOf(5f, 5f, 0.8f, 8f, 8f, 0f, 0f),
            floatArrayOf(6f, 5f, 0.7f, 8f, 8f, 0f, 0f),   // neighbour of a stronger cell -> not a peak
            floatArrayOf(15f, 15f, 0.95f, 8f, 8f, 0f, 0f),
            floatArrayOf(2f, 12f, 0.2f, 8f, 8f, 0f, 0f),   // below 0.3
        )
        val d = WideFieldMalaria.decodeDetections(m, 20, 20)
        assertEquals(listOf(0.95f, 0.8f), d.map { it.score })
    }

    @Test
    fun `decode uses the same minimum box size as the Python decode and honours the cap`() {
        val d = WideFieldMalaria.decodeDetections(map(8, 8, floatArrayOf(3f, 3f, 0.9f, 0f, 0.5f, 0f, 0f)), 8, 8)
        assertEquals(4f, d[0].x1 - d[0].x0, 1e-4f) // size clamped to >= 1 grid cell (4 px)
        val many = map(30, 30, *Array(9) { i -> floatArrayOf((i * 3).toFloat(), 10f, 0.5f + i * 0.01f, 8f, 8f, 0f, 0f) })
        assertEquals(3, WideFieldMalaria.decodeDetections(many, 30, 30, maxDetections = 3).size)
    }

    /** [n] cell scores with the highest equal to [top] and all others low. */
    private fun cells(n: Int, top: Float) = FloatArray(n) { if (it == 0) top else 0.05f }

    @Test
    fun `minimum cell count is the validated 20`() {
        assertEquals(20, WideFieldMalaria.MIN_CELLS)
    }

    @Test
    fun `any cell at or above the threshold makes a dense field positive`() {
        val r = WideFieldMalaria.interpret(cells(74, 0.985f))
        assertEquals(WideFieldMalaria.FieldStatus.POSITIVE, r.status)
        assertEquals(74, r.cellsAnalysed)
        assertEquals(0.985f, r.topCellScore, 0f)
    }

    @Test
    fun `no cell above the threshold makes a dense field negative, even with many suspicious cells`() {
        val scores = FloatArray(40) { if (it < 5) 0.98f else 0.05f } // count-based rules are not used
        assertEquals(WideFieldMalaria.FieldStatus.NEGATIVE, WideFieldMalaria.interpret(scores).status)
    }

    @Test
    fun `the count gate sits exactly at 20 cells`() {
        assertEquals(WideFieldMalaria.FieldStatus.TOO_FEW_CELLS, WideFieldMalaria.interpret(cells(19, 0.5f)).status)
        assertEquals(WideFieldMalaria.FieldStatus.NEGATIVE, WideFieldMalaria.interpret(cells(20, 0.5f)).status)
        assertEquals(WideFieldMalaria.FieldStatus.POSITIVE, WideFieldMalaria.interpret(cells(20, 0.999f)).status)
    }

    @Test
    fun `too few cells is inconclusive even when a cell scores very high, and reports how many were found`() {
        val r = WideFieldMalaria.interpret(cells(12, 0.999f))
        assertEquals(WideFieldMalaria.FieldStatus.TOO_FEW_CELLS, r.status)
        assertEquals(12, r.cellsAnalysed)
        assertEquals(0.999f, r.topCellScore, 0f)
        assertNull(WideFieldMalaria.toScreeningResult(r))
    }

    @Test
    fun `no detected cells is inconclusive, never negative`() {
        val r = WideFieldMalaria.interpret(FloatArray(0))
        assertEquals(WideFieldMalaria.FieldStatus.TOO_FEW_CELLS, r.status)
        assertEquals(0, r.cellsAnalysed)
        assertNull(WideFieldMalaria.toScreeningResult(r))
    }

    @Test
    fun `the gate reproduces the validated Phase 13 result on the 163 validation fields`() {
        // Per-field (cells, top score) summary of the validation fields that matter for the gate: the sparsest one is a negative
        // field with 19 cells and top score 0.68 (verify_gate_on_fields.py: 0/163 flag decisions change, 1 answer changes).
        assertEquals(WideFieldMalaria.FieldStatus.TOO_FEW_CELLS, WideFieldMalaria.interpret(cells(19, 0.68f)).status)
        assertEquals(WideFieldMalaria.FieldStatus.NEGATIVE, WideFieldMalaria.interpret(cells(22, 0.68f)).status)
        assertEquals(WideFieldMalaria.FieldStatus.POSITIVE, WideFieldMalaria.interpret(cells(22, 0.99f)).status) // sparsest infected field: 22
    }

    @Test
    fun `screening result carries the top cell score and the standard referral wording`() {
        val pos = WideFieldMalaria.toScreeningResult(WideFieldMalaria.interpret(cells(30, 0.996f)))
        assertNotNull(pos)
        assertEquals("Positive", pos!!.resultLabel)
        assertEquals(ScreeningInterpreter.Status.POSITIVE, pos.status)
        assertEquals(99.6f, pos.confidencePercent, 0.01f)
        assertEquals("Refer for lab confirmation", pos.referralMessage)

        val neg = WideFieldMalaria.toScreeningResult(WideFieldMalaria.interpret(cells(30, 0.3f)))!!
        assertEquals("Negative", neg.resultLabel)
        assertEquals(ScreeningInterpreter.Status.NEGATIVE, neg.status)
        assertTrue(neg.referralMessage.contains("misses"))
    }
}
