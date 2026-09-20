package com.sicklescan.app.data

data class GuardrailStats(
    /** Images that went through the guardrail check and were logged: accepted screenings + rejections. */
    val checks: Int,
    val rejections: Int,
    /** Rejections where the user chose "Analyze anyway". */
    val overrides: Int,
    val rejectionPercent: Float,
)

/**
 * Pure aggregation (no Room/Android), unit-tested locally. "Checks" =
 * disease screenings that passed the guardrail and were logged
 * ([acceptedScreenings]) + rejection events. An overridden image is counted
 * once, as a rejection -- it is deliberately NOT also a logged disease
 * screening, so it never enters the disease stats.
 */
object GuardrailAggregator {
    fun compute(acceptedScreenings: Int, events: List<GuardrailEvent>): GuardrailStats {
        val rejections = events.size
        val checks = acceptedScreenings + rejections
        return GuardrailStats(
            checks = checks,
            rejections = rejections,
            overrides = events.count { it.overridden },
            rejectionPercent = if (checks == 0) 0f else rejections * 100f / checks,
        )
    }
}
