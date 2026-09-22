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
    /**
     * [GUARDRAIL_ACCEPTED] or [GUARDRAIL_OVERRIDDEN] (user chose to continue past the image-check warning) for every mode, wide-field
     * malaria included (Phase 14). [GUARDRAIL_SKIPPED] is no longer written by the app in normal use; it survives only as a legacy
     * value on sessions logged during Phase 13 (when wide-field malaria skipped the check) and as a possible value if the check
     * could not run.
     */
    val guardrailResult: String,
) {
    companion object {
        const val GUARDRAIL_ACCEPTED = "accepted"
        const val GUARDRAIL_OVERRIDDEN = "overridden"

        /** Legacy / edge-case value: the image check did not run for this session (see above). Still counted like an accepted session by the dashboard. */
        const val GUARDRAIL_SKIPPED = "skipped"
    }
}
