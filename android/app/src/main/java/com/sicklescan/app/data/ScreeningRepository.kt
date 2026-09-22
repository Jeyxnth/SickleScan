package com.sicklescan.app.data

import android.content.Context
import com.sicklescan.app.Disease

/** One condition's outcome to be logged as part of a session. */
data class LoggedResult(
    val disease: Disease,
    val result: String,
    val confidencePercent: Float,
    val referralFlag: Boolean,
    val wideField: Boolean = false,
    val cellsDetected: Int = 0,
)

/** Thin wrapper over Room -- keeps Fragments from touching the DAO/DB directly. */
class ScreeningRepository(context: Context) {

    private val dao = AppDatabase.getInstance(context).screeningDao()
    private val guardrailDao = AppDatabase.getInstance(context).guardrailDao()

    /**
     * Logs one photo interaction: a [CaptureSession] plus its 1-2 [ScreeningRecord]s,
     * written atomically. If this session followed a guardrail rejection the user
     * overrode, pass that rejection's id so it is linked to the session.
     */
    suspend fun logSession(
        guardrailResult: String,
        results: List<LoggedResult>,
        rejectionEventId: Long? = null,
    ): Long {
        require(results.size in 1..2) { "A session has 1 or 2 condition results, got ${results.size}" }
        require(results.map { it.disease }.distinct().size == results.size) { "Duplicate disease in one session" }
        val now = System.currentTimeMillis()
        val sessionId = dao.insertSessionWithRecords(
            CaptureSession(timestampMillis = now, guardrailResult = guardrailResult),
            results.map {
                ScreeningRecord(
                    sessionId = 0, // replaced with the real id inside the transaction
                    timestampMillis = now,
                    disease = it.disease.storageKey,
                    result = it.result,
                    confidencePercent = it.confidencePercent,
                    referralFlag = it.referralFlag,
                    wideField = it.wideField,
                    cellsDetected = it.cellsDetected,
                )
            },
        )
        if (rejectionEventId != null) guardrailDao.markOverridden(rejectionEventId, sessionId)
        return sessionId
    }

    suspend fun getAllRecords(): List<ScreeningRecord> = dao.getAllRecords()

    suspend fun getAllSessions(): List<CaptureSession> = dao.getAllSessions()

    /** Logs a guardrail rejection (separate table from disease screenings). Returns its row id. */
    suspend fun logGuardrailRejection(smearScore: Float): Long =
        guardrailDao.insert(
            GuardrailEvent(timestampMillis = System.currentTimeMillis(), disease = "none", smearScore = smearScore)
        )

    suspend fun getGuardrailEvents(): List<GuardrailEvent> = guardrailDao.getAll()
}
