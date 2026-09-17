package com.sicklescan.app.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CsvExporterTest {

    @Test
    fun `header is always present, even for an empty log`() {
        val csv = CsvExporter.toCsv(emptyList())
        assertEquals("timestamp,result,confidence_percent,referral_flag\n", csv)
    }

    @Test
    fun `rows are exported oldest first with one row per record`() {
        val older = ScreeningRecord(
            timestampMillis = 1_000_000L,
            result = "negative",
            confidencePercent = 88.4f,
            referralFlag = false,
        )
        val newer = ScreeningRecord(
            timestampMillis = 2_000_000L,
            result = "positive",
            confidencePercent = 91.25f,
            referralFlag = true,
        )
        // passed in newest-first, as Room's query returns them
        val csv = CsvExporter.toCsv(listOf(newer, older))
        val lines = csv.trim().split("\n")

        assertEquals(3, lines.size) // header + 2 rows
        assertTrue("older record should come first", lines[1].contains("negative"))
        assertTrue(lines[1].contains("88.4"))
        assertTrue(lines[1].endsWith(",no"))
        assertTrue("newer record should come second", lines[2].contains("positive"))
        assertTrue(lines[2].contains("91.3")) // rounded to 1 decimal
        assertTrue(lines[2].endsWith(",yes"))
    }

    @Test
    fun `values containing commas are quoted`() {
        val record = ScreeningRecord(
            timestampMillis = 1_000_000L,
            result = "positive, confirmed",
            confidencePercent = 90f,
            referralFlag = true,
        )
        val csv = CsvExporter.toCsv(listOf(record))
        assertTrue(csv.contains("\"positive, confirmed\""))
    }
}
