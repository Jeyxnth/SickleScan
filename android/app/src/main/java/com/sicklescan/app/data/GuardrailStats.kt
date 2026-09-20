package com.sicklescan.app.data

data class GuardrailStats(
    /** Photos that went through the guardrail check: accepted sessions + rejections. */
    val checks: Int,
    val rejections: Int,
    /** Rejections where the user chose to continue anyway. */
    val overrides: Int,
    val rejectionPercent: Float,
)

/**
 * Pure aggregation (no Room/Android), unit-tested locally. "Checks" =
 * sessions that passed the guardrail ([acceptedSessions]) + rejection events.
 * A photo the user overrode is counted once, as a rejection (its later
 * overridden session is not added again), and never enters the disease stats.
 */
object GuardrailAggregator {
    fun compute(acceptedSessions: Int, events: List<GuardrailEvent>): GuardrailStats {
        val rejections = events.size
        val checks = acceptedSessions + rejections
        return GuardrailStats(
            checks = checks,
            rejections = rejections,
            overrides = events.count { it.overridden },
            rejectionPercent = if (checks == 0) 0f else rejections * 100f / checks,
        )
    }
}
