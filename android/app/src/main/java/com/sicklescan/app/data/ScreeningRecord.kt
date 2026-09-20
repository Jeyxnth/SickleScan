package com.sicklescan.app.data

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
    val confidencePercent: Float,
    val referralFlag: Boolean,
)
