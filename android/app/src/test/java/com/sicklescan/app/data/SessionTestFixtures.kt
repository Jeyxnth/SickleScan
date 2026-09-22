package com.sicklescan.app.data

/** One condition's outcome for a test photo. */
internal data class Rec(
    val disease: String,
    val result: String,
    val confidence: Float = 90f,
    val wideField: Boolean = false,
    val cells: Int = 0,
)

/** Builds consistent (sessions, records) pairs for tests: each photo() is one session with 1-2 records. */
internal class SessionFixture {
    val sessions = mutableListOf<CaptureSession>()
    val records = mutableListOf<ScreeningRecord>()
    private var nextId = 1L

    fun photo(
        timestamp: Long,
        vararg results: Rec,
        guardrail: String = CaptureSession.GUARDRAIL_ACCEPTED,
    ): Long {
        val id = nextId++
        sessions += CaptureSession(id = id, timestampMillis = timestamp, guardrailResult = guardrail)
        results.forEach {
            records += ScreeningRecord(
                id = records.size + 1L,
                sessionId = id,
                timestampMillis = timestamp,
                disease = it.disease,
                result = it.result,
                confidencePercent = it.confidence,
                // Mirrors the app: an inconclusive check (too few cells) is never flagged for referral.
                referralFlag = it.result != "negative" && it.result != ScreeningRecord.RESULT_INCONCLUSIVE,
                wideField = it.wideField,
                cellsDetected = it.cells,
            )
        }
        return id
    }
}
