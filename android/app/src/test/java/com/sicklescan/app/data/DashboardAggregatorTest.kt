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

    private val sc = "sickle_cell"
    private val mal = "malaria"

    @Test
    fun `empty log gives zeroed stats for every disease`() {
        val result = DashboardAggregator.compute(emptyList(), emptyList())
        assertEquals(0, result.sessionCount)
        assertEquals(0, result.conditionChecks)
        assertEquals(0, result.overriddenSessionCount)
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
        val f = SessionFixture()
        f.photo(now, Rec(sc, "positive", 90f))
        f.photo(now, Rec(sc, "positive", 95f))
        f.photo(now, Rec(sc, "negative", 88f))
        f.photo(now, Rec(sc, "borderline", 55f))
        // A malaria record that would badly skew a blended "% positive" if not kept separate.
        f.photo(now, Rec(mal, "negative", 99f))

        val result = DashboardAggregator.compute(f.sessions, f.records, nowMillis = now)

        assertEquals(5, result.sessionCount)
        val s = stats(Disease.SICKLE_CELL, result.perDisease)
        assertEquals(4, s.total)
        assertEquals(2, s.positiveCount)
        assertEquals(1, s.negativeCount)
        assertEquals(1, s.borderlineCount)
        assertEquals(3, s.referralCount)
        assertEquals(50f, s.positivePercent, 0.01f)
        assertEquals(25f, s.negativePercent, 0.01f)
        assertEquals(25f, s.borderlinePercent, 0.01f)

        val m = stats(Disease.MALARIA, result.perDisease)
        assertEquals(1, m.total)
        assertEquals(0, m.positiveCount)
        assertEquals(1, m.negativeCount)
        assertEquals(100f, m.negativePercent, 0.01f)
    }

    @Test
    fun `a photo screened for both conditions counts once as a session but once per condition`() {
        val now = System.currentTimeMillis()
        val f = SessionFixture()
        f.photo(now, Rec(sc, "positive"), Rec(mal, "negative"))   // both
        f.photo(now, Rec(sc, "negative"), Rec(mal, "positive"))   // both
        f.photo(now, Rec(sc, "positive"), Rec(mal, "positive"))   // both
        f.photo(now, Rec(sc, "negative"))                          // sickle only
        f.photo(now, Rec(mal, "negative"))                         // malaria only

        val r = DashboardAggregator.compute(f.sessions, f.records, nowMillis = now)

        assertEquals("headline = photos, not summed checks", 5, r.sessionCount)
        assertEquals(3, r.bothConditionsSessions)
        assertEquals(8, r.conditionChecks)
        assertEquals(4, stats(Disease.SICKLE_CELL, r.perDisease).total)
        assertEquals(4, stats(Disease.MALARIA, r.perDisease).total)
        // reconciliation the dashboard prints: photos + both-photos == condition checks == sum of per-disease
        assertEquals(r.sessionCount + r.bothConditionsSessions, r.conditionChecks)
        assertEquals(r.conditionChecks, r.perDisease.sumOf { it.total })
    }

    @Test
    fun `overridden sessions are counted separately and excluded from every stat`() {
        val now = System.currentTimeMillis()
        val f = SessionFixture()
        f.photo(now, Rec(sc, "negative"))                                                     // counted
        f.photo(now, Rec(sc, "positive"), Rec(mal, "positive"), guardrail = CaptureSession.GUARDRAIL_OVERRIDDEN)

        val r = DashboardAggregator.compute(f.sessions, f.records, nowMillis = now)

        assertEquals(1, r.sessionCount)
        assertEquals(1, r.overriddenSessionCount)
        assertEquals(1, r.conditionChecks)
        val s = stats(Disease.SICKLE_CELL, r.perDisease)
        assertEquals(1, s.total)
        assertEquals(0, s.positiveCount) // the overridden positive did not leak in
        assertEquals(0, stats(Disease.MALARIA, r.perDisease).total)
        assertEquals(0, s.dailyCounts.sumOf { it.count })
    }

    @Test
    fun `daily counts only include positive screenings for that disease, bucketed by day and zero-filled`() {
        val now = System.currentTimeMillis()
        val f = SessionFixture()
        f.photo(dayMillis(0, now), Rec(sc, "positive"))
        f.photo(dayMillis(0, now), Rec(sc, "negative"))
        f.photo(dayMillis(2, now), Rec(sc, "positive"))
        f.photo(dayMillis(2, now), Rec(sc, "borderline", 55f))
        f.photo(dayMillis(0, now), Rec(mal, "positive"))

        val result = DashboardAggregator.compute(f.sessions, f.records, nowMillis = now, windowDays = 7)

        val s = stats(Disease.SICKLE_CELL, result.perDisease)
        assertEquals(7, s.dailyCounts.size)
        assertEquals(1, s.dailyCounts.last().count) // today; its negative is excluded
        assertEquals(1, s.dailyCounts[s.dailyCounts.size - 1 - 2].count) // 2 days ago
        assertEquals(2, s.dailyCounts.sumOf { it.count }) // negative + borderline not counted

        val m = stats(Disease.MALARIA, result.perDisease)
        assertEquals(1, m.dailyCounts.sumOf { it.count }) // sickle cell's counts don't leak in
    }

    @Test
    fun `records older than the window are excluded from daily counts but not totals`() {
        val now = System.currentTimeMillis()
        val f = SessionFixture()
        f.photo(dayMillis(30, now), Rec(sc, "positive"))

        val result = DashboardAggregator.compute(f.sessions, f.records, nowMillis = now, windowDays = 7)
        val s = stats(Disease.SICKLE_CELL, result.perDisease)

        assertEquals(1, s.total) // still counted in the overall total
        assertEquals(0, s.dailyCounts.sumOf { it.count }) // but outside the 7-day trend window
    }
}
