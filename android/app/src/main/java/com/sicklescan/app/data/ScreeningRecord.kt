package com.sicklescan.app.data

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * One logged screening. No image is ever stored here (privacy) -- just the
 * result and confidence, matching exactly what the result screen showed.
 */
@Entity(tableName = "screening_records")
data class ScreeningRecord(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val timestampMillis: Long,
    /** Disease.storageKey, e.g. "sickle_cell" / "malaria" -- which model produced this. */
    val disease: String,
    /** "positive" / "borderline" / "negative" -- matches ScreeningInterpreter.Status. */
    val result: String,
    val confidencePercent: Float,
    val referralFlag: Boolean,
)
