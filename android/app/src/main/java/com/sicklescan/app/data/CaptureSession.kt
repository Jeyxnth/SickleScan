package com.sicklescan.app.data

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * One photo interaction: a single captured/selected image that went through the
 * guardrail once and was then screened for 1 or 2 conditions. Each condition's
 * outcome is a [ScreeningRecord] whose sessionId is a real foreign key to [id].
 * No image is stored.
 */
@Entity(tableName = "capture_sessions")
data class CaptureSession(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val timestampMillis: Long,
    /** [GUARDRAIL_ACCEPTED] or [GUARDRAIL_OVERRIDDEN] (user chose to continue past the image-check warning). */
    val guardrailResult: String,
) {
    companion object {
        const val GUARDRAIL_ACCEPTED = "accepted"
        const val GUARDRAIL_OVERRIDDEN = "overridden"
    }
}
