package com.sicklescan.app.data

import androidx.room.Dao
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.Insert
import androidx.room.PrimaryKey
import androidx.room.Query

/**
 * One guardrail rejection ("this doesn't look like a blood smear"). Kept in
 * its own table, NOT in screening_records, so rejections never mix into the
 * disease positive/negative/borderline stats. No image is stored.
 */
@Entity(
    tableName = "guardrail_events",
    foreignKeys = [
        ForeignKey(
            entity = CaptureSession::class,
            parentColumns = ["id"],
            childColumns = ["sessionId"],
            onDelete = ForeignKey.SET_NULL,
        ),
    ],
    indices = [Index("sessionId")],
)
data class GuardrailEvent(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val timestampMillis: Long,
    /** Unused since Phase 8 (the disease is chosen after the guardrail check); "none" for new rows. */
    val disease: String,
    /** The guardrail's P(smear) for the rejected image, in [0,1]. */
    val smearScore: Float,
    /** True if the user chose to continue past the warning. */
    val overridden: Boolean = false,
    /** The overridden session this rejection led to (set when the user continues); null otherwise. */
    val sessionId: Long? = null,
)

@Dao
interface GuardrailDao {
    @Insert
    suspend fun insert(event: GuardrailEvent): Long

    @Query("UPDATE guardrail_events SET overridden = 1, sessionId = :sessionId WHERE id = :id")
    suspend fun markOverridden(id: Long, sessionId: Long)

    @Query("SELECT * FROM guardrail_events ORDER BY timestampMillis DESC")
    suspend fun getAll(): List<GuardrailEvent>
}
