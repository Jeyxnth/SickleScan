package com.sicklescan.app.data

import com.sicklescan.app.Disease
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Locale

data class DailyCount(val label: String, val count: Int)

data class DiseaseStats(
    val disease: Disease,
    val total: Int,
    val positiveCount: Int,
    val negativeCount: Int,
    val borderlineCount: Int,
    val referralCount: Int,
    val positivePercent: Float,
    val negativePercent: Float,
    val borderlinePercent: Float,
    /** Cases flagged for lab confirmation (positive + borderline) per day, last N days, oldest first. */
    val dailyCounts: List<DailyCount>,
)

data class DashboardStats(
    /** Headline: accepted sessions = photos analyzed and counted in the stats. */
    val sessionCount: Int,
    /** Of [sessionCount], how many were screened for both conditions. */
    val bothConditionsSessions: Int,
    /** Sessions the user continued past the image-check warning: saved, but excluded from ALL stats here. */
    val overriddenSessionCount: Int,
    /** Condition checks in counted sessions == sum of perDisease totals (== sessionCount + bothConditionsSessions). */
    val conditionChecks: Int,
    /** Wide-field malaria checks that returned no verdict (too few cells): saved and exported, but not a screening outcome,
     * so they are left out of every count above. */
    val inconclusiveChecks: Int = 0,
    /** One entry per [Disease], in enum order, even when that disease has 0 screenings. */
    val perDisease: List<DiseaseStats>,
)

/**
 * Pure aggregation over the local screening log -- no Room/Android
 * dependency, so it's a real, runnable local unit test (see
 * DashboardAggregatorTest), unlike the on-device inference path.
 *
 * Stats are computed PER DISEASE, not blended into one set of numbers:
 * mixing sickle cell and malaria positives into a single "% positive"
 * would be a meaningless (and misleading) figure once the app screens
 * for more than one condition.
 */
object DashboardAggregator {

    private const val DEFAULT_WINDOW_DAYS = 7

    fun compute(
        sessions: List<CaptureSession>,
        records: List<ScreeningRecord>,
        nowMillis: Long = System.currentTimeMillis(),
        windowDays: Int = DEFAULT_WINDOW_DAYS,
    ): DashboardStats {
        // Only sessions that passed the guardrail count. Overridden sessions are saved (and exported) but reported
        // separately, so a photo the guardrail flagged can't move the positive/negative percentages. Legacy sessions
        // logged with GUARDRAIL_SKIPPED (Phase 13, wide-field malaria before the guardrail was retrained) count like accepted ones.
        // Records whose session isn't a counted one are ignored.
        val countedSessionIds = sessions
            .filter { it.guardrailResult == CaptureSession.GUARDRAIL_ACCEPTED || it.guardrailResult == CaptureSession.GUARDRAIL_SKIPPED }
            .map { it.id }
            .toSet()
        val inconclusive = records.filter { it.sessionId in countedSessionIds && it.result == ScreeningRecord.RESULT_INCONCLUSIVE }
        // A session with only an inconclusive check has no outcome to count as a photo analysed; sessions that also
        // produced a real result are counted as before.
        val counted = records.filter { it.sessionId in countedSessionIds && it.result != ScreeningRecord.RESULT_INCONCLUSIVE }
        val outcomeSessionIds = counted.map { it.sessionId }.toSet()

        val perDisease = Disease.entries.map { disease ->
            computeForDisease(disease, counted.filter { it.disease == disease.storageKey }, nowMillis, windowDays)
        }
        val bothConditions = counted.groupBy { it.sessionId }.count { (_, rs) -> rs.map { it.disease }.distinct().size == 2 }
        return DashboardStats(
            sessionCount = outcomeSessionIds.size,
            bothConditionsSessions = bothConditions,
            overriddenSessionCount = sessions.count { it.guardrailResult == CaptureSession.GUARDRAIL_OVERRIDDEN },
            conditionChecks = counted.size,
            inconclusiveChecks = inconclusive.size,
            perDisease = perDisease,
        )
    }

    private fun computeForDisease(
        disease: Disease,
        records: List<ScreeningRecord>,
        nowMillis: Long,
        windowDays: Int,
    ): DiseaseStats {
        val total = records.size
        val positiveCount = records.count { it.result == "positive" }
        val negativeCount = records.count { it.result == "negative" }
        val borderlineCount = records.count { it.result == "borderline" }
        val referralCount = records.count { it.referralFlag }

        fun percentOf(count: Int) = if (total == 0) 0f else count * 100f / total

        return DiseaseStats(
            disease = disease,
            total = total,
            positiveCount = positiveCount,
            negativeCount = negativeCount,
            borderlineCount = borderlineCount,
            referralCount = referralCount,
            positivePercent = percentOf(positiveCount),
            negativePercent = percentOf(negativeCount),
            borderlinePercent = percentOf(borderlineCount),
            // Same definition as referralCount, so the chart total always matches the "flagged" figure above it.
            dailyCounts = dailyCounts(records.filter { it.referralFlag }, nowMillis, windowDays),
        )
    }

    /** One bucket per calendar day (device-local timezone) for the last
     * [windowDays] days ending today, oldest first, zero-filled. */
    private fun dailyCounts(records: List<ScreeningRecord>, nowMillis: Long, windowDays: Int): List<DailyCount> {
        val dayFormat = SimpleDateFormat("MM/dd", Locale.US)

        fun startOfDay(millis: Long): Long =
            Calendar.getInstance().apply {
                timeInMillis = millis
                set(Calendar.HOUR_OF_DAY, 0)
                set(Calendar.MINUTE, 0)
                set(Calendar.SECOND, 0)
                set(Calendar.MILLISECOND, 0)
            }.timeInMillis

        val todayStart = startOfDay(nowMillis)
        val countsByDayStart = records
            .groupingBy { startOfDay(it.timestampMillis) }
            .eachCount()

        val cal = Calendar.getInstance().apply { timeInMillis = todayStart }
        cal.add(Calendar.DAY_OF_YEAR, -(windowDays - 1))

        return (0 until windowDays).map { _ ->
            val dayStart = cal.timeInMillis
            val count = countsByDayStart[dayStart] ?: 0
            val label = dayFormat.format(dayStart)
            cal.add(Calendar.DAY_OF_YEAR, 1)
            DailyCount(label = label, count = count)
        }
    }
}
