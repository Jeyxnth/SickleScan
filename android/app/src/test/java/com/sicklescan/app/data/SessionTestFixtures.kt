package com.sicklescan.app.data

/** One condition's outcome for a test photo. */
internal data class Rec(val disease: String, val result: String, val confidence: Float = 90f)

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
                referralFlag = it.result != "negative",
            )
        }
        return id
    }
}
