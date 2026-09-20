package com.sicklescan.app

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ResultsPresentationTest {

    @Test
    fun `warning shows whenever both conditions' results are displayed, every time`() {
        // Same rule evaluated repeatedly (each Analyze run): always true for two conditions.
        repeat(3) {
            assertTrue(ResultsPresentation.showCrossDomainWarning(listOf(Disease.SICKLE_CELL, Disease.MALARIA)))
            assertTrue(ResultsPresentation.showCrossDomainWarning(listOf(Disease.MALARIA, Disease.SICKLE_CELL)))
        }
    }

    @Test
    fun `warning is not shown for a single condition or nothing`() {
        assertFalse(ResultsPresentation.showCrossDomainWarning(listOf(Disease.SICKLE_CELL)))
        assertFalse(ResultsPresentation.showCrossDomainWarning(listOf(Disease.MALARIA)))
        assertFalse(ResultsPresentation.showCrossDomainWarning(emptyList()))
    }

    @Test
    fun `the same condition twice is not treated as two conditions`() {
        assertFalse(ResultsPresentation.showCrossDomainWarning(listOf(Disease.MALARIA, Disease.MALARIA)))
    }
}
