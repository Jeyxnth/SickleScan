package com.sicklescan.app.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CsvExporterTest {

    private val header = "session_id,timestamp,disease,result,confidence_percent,referral_flag,image_check,input_mode,cells_detected\n"

    @Test
    fun `header is always present, even for an empty log`() {
        assertEquals(header, CsvExporter.toCsv(emptyList(), emptyList()))
    }

    @Test
    fun `two conditions from one photo share a session id and timestamp, rows adjacent`() {
        val f = SessionFixture()
        f.photo(1_000_000L, Rec("sickle_cell", "negative", 88.4f))
        f.photo(2_000_000L, Rec("sickle_cell", "positive", 96.5f), Rec("malaria", "negative", 91.25f))

        // passed newest-first, as Room's query returns them
        val csv = CsvExporter.toCsv(f.sessions.reversed(), f.records.reversed())
        val lines = csv.trim().split("\n")

        assertEquals(4, lines.size) // header + 3 rows
        assertTrue("oldest session first", lines[1].startsWith("1,") && lines[1].contains("sickle_cell") && lines[1].contains("88.4"))
        assertTrue(lines[1].endsWith(",no,accepted,image,0"))
        // the two rows from the second photo carry the same session_id and the same timestamp
        val a = lines[2].split(",")
        val b = lines[3].split(",")
        assertEquals("2", a[0])
        assertEquals(a[0], b[0])
        assertEquals(a[1], b[1])
        assertEquals("malaria/sickle_cell rows adjacent, ordered by disease", listOf("malaria", "sickle_cell"), listOf(a[2], b[2]).sorted())
        assertTrue(lines[3].contains("91.3") || lines[2].contains("91.3")) // rounded to 1 decimal
    }

    @Test
    fun `overridden sessions are flagged in the image_check column`() {
        val f = SessionFixture()
        f.photo(1_000_000L, Rec("malaria", "positive"), guardrail = CaptureSession.GUARDRAIL_OVERRIDDEN)
        val csv = CsvExporter.toCsv(f.sessions, f.records)
        assertTrue(csv.trim().split("\n")[1].endsWith(",overridden,image,0"))
    }

    @Test
    fun `wide-field malaria rows are marked and carry the number of cells scored`() {
        val f = SessionFixture()
        f.photo(1_000_000L, Rec("malaria", "positive", 99.6f, wideField = true, cells = 74), Rec("sickle_cell", "negative", 80f))
        val rows = CsvExporter.toCsv(f.sessions, f.records).trim().split("\n").drop(1)
        val malaria = rows.first { it.contains(",malaria,") }
        val sickle = rows.first { it.contains(",sickle_cell,") }
        assertTrue(malaria, malaria.endsWith(",yes,accepted,field,74"))
        assertTrue(sickle, sickle.endsWith(",no,accepted,image,0"))
    }

    @Test
    fun `sessions that never ran the image check are distinguishable in the image_check column`() {
        val f = SessionFixture()
        f.photo(1_000_000L, Rec("malaria", "positive", 99.6f, wideField = true, cells = 74), guardrail = CaptureSession.GUARDRAIL_SKIPPED)
        f.photo(2_000_000L, Rec("malaria", "negative"))
        val rows = CsvExporter.toCsv(f.sessions, f.records).trim().split("\n").drop(1)
        assertTrue(rows[0], rows[0].endsWith(",skipped,field,74"))
        assertTrue(rows[1], rows[1].endsWith(",accepted,image,0"))
    }

    @Test
    fun `a count-gate inconclusive is logged as its own row with the cell count and a skipped image check`() {
        val f = SessionFixture()
        f.photo(1_000_000L, Rec("malaria", ScreeningRecord.RESULT_INCONCLUSIVE, 68f, wideField = true, cells = 12), guardrail = CaptureSession.GUARDRAIL_SKIPPED)
        val row = CsvExporter.toCsv(f.sessions, f.records).trim().split("\n")[1]
        assertTrue(row, row.contains(",malaria,inconclusive,68.0,no,"))
        assertTrue(row, row.endsWith(",skipped,field,12"))
    }

    @Test
    fun `values containing commas are quoted`() {
        val f = SessionFixture()
        f.photo(1_000_000L, Rec("sickle_cell", "positive, confirmed"))
        val csv = CsvExporter.toCsv(f.sessions, f.records)
        assertTrue(csv.contains("\"positive, confirmed\""))
    }
}
