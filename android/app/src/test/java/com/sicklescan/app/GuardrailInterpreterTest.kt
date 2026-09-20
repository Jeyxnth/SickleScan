package com.sicklescan.app

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GuardrailInterpreterTest {

    @Test
    fun `confident smear is accepted`() {
        assertTrue(GuardrailInterpreter.looksLikeSmear(0.999f))
    }

    @Test
    fun `confident non-smear is rejected`() {
        assertFalse(GuardrailInterpreter.looksLikeSmear(0.001f))
    }

    @Test
    fun `exactly at the threshold is accepted, just below is rejected`() {
        assertTrue(GuardrailInterpreter.looksLikeSmear(GuardrailInterpreter.REJECT_BELOW))
        assertFalse(GuardrailInterpreter.looksLikeSmear(GuardrailInterpreter.REJECT_BELOW - 0.001f))
    }
}
