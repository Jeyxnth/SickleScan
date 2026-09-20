package com.sicklescan.app.data

import com.sicklescan.app.Disease
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

    private fun stats(disease: Disease, all: List<DiseaseStats>) =
        all.first { it.disease == disease }

    @Test
    fun `empty log gives zeroed stats for every disease`() {
        val result = DashboardAggregator.compute(emptyList())
        assertEquals(0, result.totalAllDiseases)
        assertEquals(Disease.entries.size, result.perDisease.size)
        result.perDisease.forEach { d ->
            assertEquals(0, d.total)
            assertEquals(0f, d.positivePercent, 0.01f)
            assertEquals(7, d.dailyCounts.size)
            assertEquals(0, d.dailyCounts.sumOf { it.count })
        }
    }

    @Test
    fun `counts and percentages are scoped to their own disease, not blended`() {
        val now = System.currentTimeMillis()
        val records = listOf(
            ScreeningRecord(timestampMillis = now, disease = "sickle_cell", result = "positive", confidencePercent = 90f, referralFlag = true),
            ScreeningRecord(timestampMillis = now, disease = "sickle_cell", result = "positive", confidencePercent = 95f, referralFlag = true),
            ScreeningRecord(timestampMillis = now, disease = "sickle_cell", result = "negative", confidencePercent = 88f, referralFlag = false),
            ScreeningRecord(timestampMillis = now, disease = "sickle_cell", result = "borderline", confidencePercent = 55f, referralFlag = true),
            // A malaria record that would badly skew a blended "% positive" if not kept separate.
            ScreeningRecord(timestampMillis = now, disease = "malaria", result = "negative", confidencePercent = 99f, referralFlag = false),
        )

        val result = DashboardAggregator.compute(records, nowMillis = now)

        assertEquals(5, result.totalAllDiseases)

        val sc = stats(Disease.SICKLE_CELL, result.perDisease)
        assertEquals(4, sc.total)
        assertEquals(2, sc.positiveCount)
        assertEquals(1, sc.negativeCount)
        assertEquals(1, sc.borderlineCount)
        assertEquals(3, sc.referralCount)
        assertEquals(50f, sc.positivePercent, 0.01f)
        assertEquals(25f, sc.negativePercent, 0.01f)
        assertEquals(25f, sc.borderlinePercent, 0.01f)

        val mal = stats(Disease.MALARIA, result.perDisease)
        assertEquals(1, mal.total)
        assertEquals(0, mal.positiveCount)
        assertEquals(1, mal.negativeCount)
        assertEquals(100f, mal.negativePercent, 0.01f)
    }

    @Test
    fun `daily counts only include positive screenings for that disease, bucketed by day and zero-filled`() {
        val now = System.currentTimeMillis()
        val records = listOf(
            ScreeningRecord(timestampMillis = dayMillis(0, now), disease = "sickle_cell", result = "positive", confidencePercent = 90f, referralFlag = true),
            ScreeningRecord(timestampMillis = dayMillis(0, now), disease = "sickle_cell", result = "negative", confidencePercent = 90f, referralFlag = false),
            ScreeningRecord(timestampMillis = dayMillis(2, now), disease = "sickle_cell", result = "positive", confidencePercent = 90f, referralFlag = true),
            ScreeningRecord(timestampMillis = dayMillis(2, now), disease = "sickle_cell", result = "borderline", confidencePercent = 55f, referralFlag = true),
            ScreeningRecord(timestampMillis = dayMillis(0, now), disease = "malaria", result = "positive", confidencePercent = 90f, referralFlag = true),
        )

        val result = DashboardAggregator.compute(records, nowMillis = now, windowDays = 7)

        val sc = stats(Disease.SICKLE_CELL, result.perDisease)
        assertEquals(7, sc.dailyCounts.size)
        assertEquals(1, sc.dailyCounts.last().count) // today; its negative is excluded
        assertEquals(1, sc.dailyCounts[sc.dailyCounts.size - 1 - 2].count) // 2 days ago
        assertEquals(2, sc.dailyCounts.sumOf { it.count }) // negative + borderline not counted

        val mal = stats(Disease.MALARIA, result.perDisease)
        assertEquals(1, mal.dailyCounts.sumOf { it.count }) // sickle cell's counts don't leak in
    }

    @Test
    fun `records older than the window are excluded from daily counts but not totals`() {
        val now = System.currentTimeMillis()
        val records = listOf(
            ScreeningRecord(timestampMillis = dayMillis(30, now), disease = "sickle_cell", result = "positive", confidencePercent = 90f, referralFlag = true),
        )

        val result = DashboardAggregator.compute(records, nowMillis = now, windowDays = 7)
        val sc = stats(Disease.SICKLE_CELL, result.perDisease)

        assertEquals(1, sc.total) // still counted in the overall total
        assertEquals(0, sc.dailyCounts.sumOf { it.count }) // but outside the 7-day trend window
    }
}
