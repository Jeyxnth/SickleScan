package com.sicklescan.app.data

import org.junit.Assert.assertEquals
import org.junit.Test

class GuardrailAggregatorTest {

    private fun event(overridden: Boolean = false) =
        GuardrailEvent(timestampMillis = 0L, disease = "none", smearScore = 0.02f, overridden = overridden)

    @Test
    fun `no activity gives zeros and no divide-by-zero`() {
        val s = GuardrailAggregator.compute(acceptedSessions = 0, events = emptyList())
        assertEquals(0, s.checks)
        assertEquals(0, s.rejections)
        assertEquals(0f, s.rejectionPercent, 0.0001f)
    }

    @Test
    fun `rejection rate is rejections over accepted sessions plus rejected`() {
        // 6 accepted sessions + 2 rejections = 8 checks, 25% rejected
        val s = GuardrailAggregator.compute(acceptedSessions = 6, events = listOf(event(), event()))
        assertEquals(8, s.checks)
        assertEquals(2, s.rejections)
        assertEquals(25f, s.rejectionPercent, 0.0001f)
    }

    @Test
    fun `overrides are counted but do not add extra checks`() {
        val s = GuardrailAggregator.compute(acceptedSessions = 3, events = listOf(event(overridden = true), event()))
        assertEquals(5, s.checks)
        assertEquals(2, s.rejections)
        assertEquals(1, s.overrides)
    }
}
