package com.sicklescan.app.data

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * One condition's screening outcome within a [CaptureSession]. A session owns
 * 1 or 2 of these (one per condition the user selected for that photo). The
 * sessionId is a real foreign key (CASCADE: deleting a session deletes its
 * records; a record can't exist without its session), not a timestamp match.
 * No image is ever stored here (privacy) -- just the result and confidence,
 * matching exactly what the result screen showed.
 */
@Entity(
    tableName = "screening_records",
    foreignKeys = [
        ForeignKey(
            entity = CaptureSession::class,
            parentColumns = ["id"],
            childColumns = ["sessionId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("sessionId")],
)
data class ScreeningRecord(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val sessionId: Long,
    val timestampMillis: Long,
    /** Disease.storageKey, e.g. "sickle_cell" / "malaria" -- which model produced this. */
    val disease: String,
    /** "positive" / "borderline" / "negative" -- matches ScreeningInterpreter.Status. */
    val result: String,
    /** Single-cell path: model confidence in the stated result. Wide-field path: the highest single-cell score. */
    val confidencePercent: Float,
    val referralFlag: Boolean,
    /** Phase 13: true for a wide-field malaria check (detect cells, classify each, any-cell rule); false for a direct
     * whole-image classification. Added in DB v5 with a default, so older rows read as false. */
    @ColumnInfo(defaultValue = "0") val wideField: Boolean = false,
    /** Phase 13: cells the detector found and the classifier scored (0 for direct classification). */
    @ColumnInfo(defaultValue = "0") val cellsDetected: Int = 0,
) {
    companion object {
        /**
         * Phase 13: a wide-field check that returned no verdict because fewer than WideFieldMalaria.MIN_CELLS cells were
         * detected (cellsDetected says how many, 0 included). Logged so it is visible in the data; it is not a screening
         * outcome, so the dashboard keeps it out of the positive / negative / borderline statistics.
         */
        const val RESULT_INCONCLUSIVE = "inconclusive"
    }
}
