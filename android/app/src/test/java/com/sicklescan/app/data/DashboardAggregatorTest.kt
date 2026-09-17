package com.sicklescan.app.data

import org.junit.Assert.assertEquals
import org.junit.Test
import java.util.Calendar

class DashboardAggregatorTest {

    private fun dayMillis(daysAgo: Int, referenceMillis: Long): Long {
        val cal = Calendar.getInstance().apply {
            timeInMillis = referenceMillis
            add(Calendar.DAY_OF_YEAR, -daysAgo)
            set(Calendar.HOUR_OF_DAY, 10) // arbitrary time within the day
        }
        return cal.timeInMillis
    }

    @Test
    fun `empty log gives zeroed stats`() {
        val stats = DashboardAggregator.compute(emptyList())
        assertEquals(0, stats.total)
        assertEquals(0f, stats.positivePercent, 0.01f)
        assertEquals(7, stats.dailyCounts.size)
        assertEquals(0, stats.dailyCounts.sumOf { it.count })
    }

    @Test
    fun `counts and percentages match a mixed log`() {
        val now = System.currentTimeMillis()
        val records = listOf(
            ScreeningRecord(timestampMillis = now, result = "positive", confidencePercent = 90f, referralFlag = true),
            ScreeningRecord(timestampMillis = now, result = "positive", confidencePercent = 95f, referralFlag = true),
            ScreeningRecord(timestampMillis = now, result = "negative", confidencePercent = 88f, referralFlag = false),
            ScreeningRecord(timestampMillis = now, result = "borderline", confidencePercent = 55f, referralFlag = true),
        )

        val stats = DashboardAggregator.compute(records, nowMillis = now)

        assertEquals(4, stats.total)
        assertEquals(2, stats.positiveCount)
        assertEquals(1, stats.negativeCount)
        assertEquals(1, stats.borderlineCount)
        assertEquals(3, stats.referralCount) // 2 positive + 1 borderline
        assertEquals(50f, stats.positivePercent, 0.01f)
        assertEquals(25f, stats.negativePercent, 0.01f)
        assertEquals(25f, stats.borderlinePercent, 0.01f)
    }

    @Test
    fun `daily counts bucket records into the right day and zero-fill the rest`() {
        val now = System.currentTimeMillis()
        val records = listOf(
            ScreeningRecord(timestampMillis = dayMillis(0, now), result = "positive", confidencePercent = 90f, referralFlag = true),
            ScreeningRecord(timestampMillis = dayMillis(0, now), result = "negative", confidencePercent = 90f, referralFlag = false),
            ScreeningRecord(timestampMillis = dayMillis(2, now), result = "negative", confidencePercent = 90f, referralFlag = false),
        )

        val stats = DashboardAggregator.compute(records, nowMillis = now, windowDays = 7)

        assertEquals(7, stats.dailyCounts.size)
        assertEquals(2, stats.dailyCounts.last().count) // today = index 6 (oldest-first list)
        assertEquals(1, stats.dailyCounts[stats.dailyCounts.size - 1 - 2].count) // 2 days ago
        assertEquals(3, stats.dailyCounts.sumOf { it.count })
    }

    @Test
    fun `records older than the window are excluded from daily counts but not totals`() {
        val now = System.currentTimeMillis()
        val records = listOf(
            ScreeningRecord(timestampMillis = dayMillis(30, now), result = "positive", confidencePercent = 90f, referralFlag = true),
        )

        val stats = DashboardAggregator.compute(records, nowMillis = now, windowDays = 7)

        assertEquals(1, stats.total) // still counted in the overall total
        assertEquals(0, stats.dailyCounts.sumOf { it.count }) // but outside the 7-day trend window
    }
}
