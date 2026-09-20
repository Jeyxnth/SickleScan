package com.sicklescan.app.data

import android.content.Context
import com.sicklescan.app.Disease

/** Thin wrapper over Room -- keeps Fragments from touching the DAO/DB directly. */
class ScreeningRepository(context: Context) {

    private val dao = AppDatabase.getInstance(context).screeningDao()
    private val guardrailDao = AppDatabase.getInstance(context).guardrailDao()

    suspend fun logScreening(disease: Disease, result: String, confidencePercent: Float, referralFlag: Boolean) {
        dao.insert(
            ScreeningRecord(
                timestampMillis = System.currentTimeMillis(),
                disease = disease.storageKey,
                result = result,
                confidencePercent = confidencePercent,
                referralFlag = referralFlag,
            )
        )
    }

    suspend fun getAllRecords(): List<ScreeningRecord> = dao.getAll()

    /** Logs a guardrail rejection (separate table from disease screenings). Returns its row id. */
    suspend fun logGuardrailRejection(disease: Disease, smearScore: Float): Long =
        guardrailDao.insert(
            GuardrailEvent(timestampMillis = System.currentTimeMillis(), disease = disease.storageKey, smearScore = smearScore)
        )

    suspend fun markGuardrailOverridden(eventId: Long) = guardrailDao.markOverridden(eventId)

    suspend fun getGuardrailEvents(): List<GuardrailEvent> = guardrailDao.getAll()
}
