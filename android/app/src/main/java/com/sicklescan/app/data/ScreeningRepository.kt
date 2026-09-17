package com.sicklescan.app.data

import android.content.Context

/** Thin wrapper over Room -- keeps Fragments from touching the DAO/DB directly. */
class ScreeningRepository(context: Context) {

    private val dao = AppDatabase.getInstance(context).screeningDao()

    suspend fun logScreening(result: String, confidencePercent: Float, referralFlag: Boolean) {
        dao.insert(
            ScreeningRecord(
                timestampMillis = System.currentTimeMillis(),
                result = result,
                confidencePercent = confidencePercent,
                referralFlag = referralFlag,
            )
        )
    }

    suspend fun getAllRecords(): List<ScreeningRecord> = dao.getAll()
}
