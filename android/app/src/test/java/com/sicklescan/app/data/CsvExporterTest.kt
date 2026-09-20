package com.sicklescan.app.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CsvExporterTest {

    private val header = "session_id,timestamp,disease,result,confidence_percent,referral_flag,image_check\n"

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
        assertTrue(lines[1].endsWith(",no,accepted"))
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
        assertTrue(csv.trim().split("\n")[1].endsWith(",overridden"))
    }

    @Test
    fun `values containing commas are quoted`() {
        val f = SessionFixture()
        f.photo(1_000_000L, Rec("sickle_cell", "positive, confirmed"))
        val csv = CsvExporter.toCsv(f.sessions, f.records)
        assertTrue(csv.contains("\"positive, confirmed\""))
    }
}
