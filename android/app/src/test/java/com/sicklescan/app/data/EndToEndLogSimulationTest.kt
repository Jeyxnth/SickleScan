package com.sicklescan.app.data

import com.sicklescan.app.Disease
import com.sicklescan.app.ScreeningInterpreter
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Calendar

/**
 * Simulates "run a handful of sessions" end to end using raw probabilities that
 * were independently computed in Python by running the ACTUAL bundled .tflite
 * files (see android/README.md) -- the closest thing to a device run available
 * here. Exercises the same ScreeningInterpreter -> CaptureSession/ScreeningRecord
 * -> DashboardAggregator -> CsvExporter pipeline the app uses, and prints the
 * CSV and dashboard numbers so they can be inspected by hand.
 *
 * Includes the Phase 8 case: ONE photo screened for BOTH conditions. The pair
 * below are real same-image outputs from both disease models
 * (sickle model, malaria model) for one sample image from each disease's set.
 */
class EndToEndLogSimulationTest {

    // Sickle-cell-only sessions: fileName -> raw positive probability from sicklescan_model.tflite.
    private val sickleOnly = linkedMapOf(
        "pos_325.jpg" to 0.9849f,
        "pos_231.jpg" to 0.9930f,
        "pos_20.jpg" to 0.9646f,
        "pos_64.jpg" to 0.9914f,
        "pos_213.jpg" to 0.9793f,
        "pos_155.jpg" to 0.9910f,
        "pos_95.jpg" to 0.9985f,
        "neg_46.jpg" to 0.0178f,
        "pos_21.jpg" to 0.7256f,
        "neg_140.jpg" to 0.1248f,
        "pos_393.jpg" to 0.9999f,
    )

    // Both-conditions sessions: ONE image, run through BOTH shipped models (sickle P, malaria P).
    // Malaria values are from the Phase 10 BBBC041-trained model (NIH-style crops score low there --
    // a real limitation of that model on NIH-style images, see model_output/malaria_bbbc041/).
    private val bothConditions = listOf(
        Triple("pos_20.jpg (sickle set)", 0.9646f, 0.0945f),
        Triple("parasitized_1.png (NIH-style malaria crop)", 0.8061f, 0.0288f),
    )

    // Malaria-only sessions: real outputs of the shipped model on three held-out BBBC041 demo crops
    // (a confident infected, a low-confidence infected that lands in Borderline, a confident uninfected).
    private val malariaOnly = linkedMapOf(
        "infected_1_ring.png" to 0.9912f,
        "infected_5_trophozoite.png" to 0.5343f,
        "uninfected_1_redbloodcell.png" to 0.0010f,
    )

    private fun rec(disease: Disease, p: Float): Rec {
        val r = ScreeningInterpreter.interpret(disease, p)
        return Rec(disease.storageKey, r.status.name.lowercase(), r.confidencePercent)
    }

    @Test
    fun `sessions, per-disease stats and CSV linkage all reconcile`() {
        val now = System.currentTimeMillis()
        val f = SessionFixture()
        var day = 0
        fun ts() = Calendar.getInstance().apply { timeInMillis = now; add(Calendar.DAY_OF_YEAR, -(day++ % 4)) }.timeInMillis

        sickleOnly.values.forEach { p -> f.photo(ts(), rec(Disease.SICKLE_CELL, p)) }
        bothConditions.forEach { (_, sickleP, malariaP) ->
            f.photo(ts(), rec(Disease.SICKLE_CELL, sickleP), rec(Disease.MALARIA, malariaP))
        }
        malariaOnly.values.forEach { p -> f.photo(ts(), rec(Disease.MALARIA, p)) }
        // One photo the guardrail flagged that the user continued past: saved + exported, excluded from stats.
        f.photo(ts(), rec(Disease.SICKLE_CELL, 0.99f), guardrail = CaptureSession.GUARDRAIL_OVERRIDDEN)

        val stats = DashboardAggregator.compute(f.sessions, f.records, nowMillis = now)

        // 11 sickle-only + 2 both-condition + 3 malaria-only photos = 16 sessions (the overridden one is separate).
        assertEquals(16, stats.sessionCount)
        assertEquals(2, stats.bothConditionsSessions)
        assertEquals(1, stats.overriddenSessionCount)
        assertEquals(18, stats.conditionChecks) // 11 + 2*2 + 3
        assertEquals(stats.sessionCount + stats.bothConditionsSessions, stats.conditionChecks)

        val sc = stats.perDisease.first { it.disease == Disease.SICKLE_CELL }
        assertEquals(13, sc.total) // 11 + 2 (overridden excluded)
        assertEquals(11, sc.positiveCount) // 9 of the 11 + both pair photos positive (0.9646, 0.8061)
        assertEquals(2, sc.negativeCount)
        val mal = stats.perDisease.first { it.disease == Disease.MALARIA }
        assertEquals(5, mal.total)                 // 2 from the both-condition photos + 3 malaria-only
        assertEquals(1, mal.positiveCount)         // infected_1_ring (0.9912)
        assertEquals(1, mal.borderlineCount)       // infected_5_trophozoite: 53.4% confidence < 65% ceiling
        assertEquals(3, mal.negativeCount)         // the two both-condition photos + uninfected_1
        assertEquals(stats.conditionChecks, stats.perDisease.sumOf { it.total })

        val csv = CsvExporter.toCsv(f.sessions, f.records)
        println("---- Simulated CSV export ----")
        println(csv)
        println("---- Dashboard ----")
        println(
            "sessions=${stats.sessionCount} both=${stats.bothConditionsSessions} overridden=${stats.overriddenSessionCount} " +
                "conditionChecks=${stats.conditionChecks} | sickle=${sc.total} malaria=${mal.total}"
        )

        val lines = csv.trim().split("\n")
        assertEquals("session_id,timestamp,disease,result,confidence_percent,referral_flag,image_check", lines[0])
        assertEquals(1 + 19, lines.size) // 18 counted rows + 1 overridden row
        // Session linkage: each both-condition photo yields two rows with the same session_id.
        val rowsBySession = lines.drop(1).map { it.split(",") }.groupBy { it[0] }
        assertEquals(2, rowsBySession.values.count { it.size == 2 })
        rowsBySession.values.filter { it.size == 2 }.forEach { rows ->
            assertEquals(setOf("sickle_cell", "malaria"), rows.map { it[2] }.toSet())
            assertEquals("same photo -> same timestamp", rows[0][1], rows[1][1])
        }
        assertTrue(lines.drop(1).count { it.endsWith(",overridden") } == 1)
    }

    /**
     * Cross-checks the two weekly "flagged for lab confirmation" charts against a hand tally of the raw
     * simulated results that never touches DashboardAggregator: every result is recorded
     * as (day offset, disease, status) as it is generated, then tallied independently.
     */
    @Test
    fun `weekly flagged-case charts match a hand tally of the raw simulated results`() {
        val now = System.currentTimeMillis()
        val f = SessionFixture()
        val raw = mutableListOf<Triple<Int, String, String>>() // (days ago, disease, status), counted sessions only
        var n = 0

        fun photo(overridden: Boolean, vararg probs: Pair<Disease, Float>) {
            val daysAgo = n++ % 4
            val ts = Calendar.getInstance().apply { timeInMillis = now; add(Calendar.DAY_OF_YEAR, -daysAgo) }.timeInMillis
            val recs = probs.map { (d, p) -> rec(d, p) }
            f.photo(
                ts, *recs.toTypedArray(),
                guardrail = if (overridden) CaptureSession.GUARDRAIL_OVERRIDDEN else CaptureSession.GUARDRAIL_ACCEPTED,
            )
            if (!overridden) recs.forEach { raw += Triple(daysAgo, it.disease, it.result) }
        }

        // Same data as the simulation above (+ the overridden positive that must NOT appear).
        sickleOnly.values.forEach { p -> photo(false, Disease.SICKLE_CELL to p) }
        bothConditions.forEach { (_, s, m) -> photo(false, Disease.SICKLE_CELL to s, Disease.MALARIA to m) }
        malariaOnly.values.forEach { p -> photo(false, Disease.MALARIA to p) }
        photo(true, Disease.SICKLE_CELL to 0.99f)

        val stats = DashboardAggregator.compute(f.sessions, f.records, nowMillis = now, windowDays = 7)

        for (disease in Disease.entries) {
            val chart = stats.perDisease.first { it.disease == disease }.dailyCounts
            assertEquals(7, chart.size)
            // Independent expectation: every case FLAGGED for lab confirmation (positive + borderline, not negative), this disease only.
            val expectedByDaysAgo = raw
                .filter { it.second == disease.storageKey && it.third != "negative" }
                .groupingBy { it.first }.eachCount()
            for (daysAgo in 0..6) {
                assertEquals(
                    "${disease.displayName}, $daysAgo days ago",
                    expectedByDaysAgo[daysAgo] ?: 0,
                    chart[6 - daysAgo].count, // chart is oldest-first, today last
                )
            }
            println("${disease.displayName} — flagged cases this week: " + chart.joinToString("  ") { "${it.label}:${it.count}" })
        }

        val sickleChart = stats.perDisease.first { it.disease == Disease.SICKLE_CELL }.dailyCounts
        val malariaChart = stats.perDisease.first { it.disease == Disease.MALARIA }.dailyCounts
        assertEquals("9 positives among the 11 sickle-only + 2 pair photos; overridden one excluded", 11, sickleChart.sumOf { it.count })
        assertEquals("infected_1_ring (positive) + infected_5_trophozoite (borderline) are flagged; the negatives are not", 2, malariaChart.sumOf { it.count })
        // Charts exclude negatives, so they must be smaller than the totals that include them.
        assertTrue(sickleChart.sumOf { it.count } < stats.perDisease.first { it.disease == Disease.SICKLE_CELL }.total)
    }
}
