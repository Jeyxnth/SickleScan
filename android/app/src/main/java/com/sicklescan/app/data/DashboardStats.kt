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
    /** Positive screenings per day, last N days, oldest first. */
    val dailyCounts: List<DailyCount>,
)

data class DashboardStats(
    val totalAllDiseases: Int,
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
        records: List<ScreeningRecord>,
        nowMillis: Long = System.currentTimeMillis(),
        windowDays: Int = DEFAULT_WINDOW_DAYS,
    ): DashboardStats {
        val perDisease = Disease.entries.map { disease ->
            computeForDisease(disease, records.filter { it.disease == disease.storageKey }, nowMillis, windowDays)
        }
        return DashboardStats(totalAllDiseases = records.size, perDisease = perDisease)
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
            dailyCounts = dailyCounts(records.filter { it.result == "positive" }, nowMillis, windowDays),
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
