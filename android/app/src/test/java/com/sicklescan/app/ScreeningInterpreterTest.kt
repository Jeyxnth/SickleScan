package com.sicklescan.app

import org.junit.Assert.assertEquals
import org.junit.Test

class ScreeningInterpreterTest {

    @Test
    fun `high-confidence positive is referred`() {
        val result = ScreeningInterpreter.interpret(Disease.SICKLE_CELL, 0.9f)
        assertEquals(ScreeningInterpreter.Status.POSITIVE, result.status)
        assertEquals("Positive", result.resultLabel)
        assertEquals("Refer for lab confirmation", result.referralMessage)
        assertEquals(90f, result.confidencePercent, 0.01f)
    }

    @Test
    fun `high-confidence negative is low risk`() {
        val result = ScreeningInterpreter.interpret(Disease.SICKLE_CELL, 0.1f)
        assertEquals(ScreeningInterpreter.Status.NEGATIVE, result.status)
        assertEquals("Negative", result.resultLabel)
        assertEquals("Low risk — routine monitoring", result.referralMessage)
        assertEquals(90f, result.confidencePercent, 0.01f)
    }

    @Test
    fun `just-above-threshold positive is borderline, not confidently positive`() {
        // p=0.6 -> predicted positive, but only 60% confident (below the 65% ceiling)
        val result = ScreeningInterpreter.interpret(Disease.SICKLE_CELL, 0.6f)
        assertEquals(ScreeningInterpreter.Status.BORDERLINE, result.status)
        assertEquals(60f, result.confidencePercent, 0.01f)
        assertEquals("Uncertain result — refer for lab confirmation", result.referralMessage)
    }

    @Test
    fun `just-below-threshold negative is also borderline`() {
        // p=0.4 -> predicted negative, but only 60% confident
        val result = ScreeningInterpreter.interpret(Disease.SICKLE_CELL, 0.4f)
        assertEquals(ScreeningInterpreter.Status.BORDERLINE, result.status)
        assertEquals("Negative", result.resultLabel)
        assertEquals(60f, result.confidencePercent, 0.01f)
        assertEquals("Uncertain result — refer for lab confirmation", result.referralMessage)
    }

    @Test
    fun `exactly at the decision threshold is borderline`() {
        val result = ScreeningInterpreter.interpret(Disease.SICKLE_CELL, 0.5f)
        assertEquals(ScreeningInterpreter.Status.BORDERLINE, result.status)
        assertEquals(50f, result.confidencePercent, 0.01f)
    }

    @Test
    fun `just at the borderline ceiling counts as confident, not borderline`() {
        // p=0.65 -> exactly 65% confidence, at the ceiling -- spec says "50-65%"
        // is borderline, so 65% itself is treated as confidently positive.
        val result = ScreeningInterpreter.interpret(Disease.SICKLE_CELL, 0.65f)
        assertEquals(ScreeningInterpreter.Status.POSITIVE, result.status)
        assertEquals("Refer for lab confirmation", result.referralMessage)
    }

    @Test
    fun `just under the borderline ceiling is still borderline`() {
        val result = ScreeningInterpreter.interpret(Disease.SICKLE_CELL, 0.6499f)
        assertEquals(ScreeningInterpreter.Status.BORDERLINE, result.status)
    }
}
