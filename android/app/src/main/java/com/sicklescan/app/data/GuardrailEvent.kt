package com.sicklescan.app.data

import androidx.room.Dao
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.PrimaryKey
import androidx.room.Query

/**
 * One guardrail rejection ("this doesn't look like a blood smear"). Kept in
 * its own table, NOT in screening_records, so rejections never mix into the
 * disease positive/negative/borderline stats. No image is stored.
 */
@Entity(tableName = "guardrail_events")
data class GuardrailEvent(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val timestampMillis: Long,
    /** Disease.storageKey the user had selected when the image was rejected. */
    val disease: String,
    /** The guardrail's P(smear) for the rejected image, in [0,1]. */
    val smearScore: Float,
    /** True if the user tapped "Analyze anyway" after the warning. */
    val overridden: Boolean = false,
)

@Dao
interface GuardrailDao {
    @Insert
    suspend fun insert(event: GuardrailEvent): Long

    @Query("UPDATE guardrail_events SET overridden = 1 WHERE id = :id")
    suspend fun markOverridden(id: Long)

    @Query("SELECT * FROM guardrail_events ORDER BY timestampMillis DESC")
    suspend fun getAll(): List<GuardrailEvent>
}
