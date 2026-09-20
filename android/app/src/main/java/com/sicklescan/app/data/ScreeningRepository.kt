package com.sicklescan.app.data

import android.content.Context
import com.sicklescan.app.Disease

/** Thin wrapper over Room -- keeps Fragments from touching the DAO/DB directly. */
class ScreeningRepository(context: Context) {

    private val dao = AppDatabase.getInstance(context).screeningDao()

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
}
