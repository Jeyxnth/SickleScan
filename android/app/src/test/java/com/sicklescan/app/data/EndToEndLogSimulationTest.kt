package com.sicklescan.app.data

import com.sicklescan.app.Disease
import com.sicklescan.app.ScreeningInterpreter
import org.junit.Assert.assertEquals
import org.junit.Test
import java.util.Calendar

/**
 * Simulates "run a handful of screenings" end to end using the real raw
 * probabilities already independently verified (in Python, against the
 * actual sicklescan_model.tflite) for the 11 known sample images from
 * Phases 2-3 -- the closest thing to a real device run available in this
 * environment. Exercises the exact
 * ScreeningInterpreter -> ScreeningRecord -> DashboardAggregator ->
 * CsvExporter pipeline the app uses, now with a disease tag on every
 * record, and prints the CSV so it can be inspected by hand.
 */
class EndToEndLogSimulationTest {

    // fileName -> raw positive-class probability from the .tflite model
    // (verified against the model directly; see android/README.md).
    private val knownProbabilities = linkedMapOf(
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

    @Test
    fun `simulated screening log produces correct per-disease dashboard stats and CSV`() {
        val now = System.currentTimeMillis()
        var day = 0
        val records = knownProbabilities.map { (_, probability) ->
            val result = ScreeningInterpreter.interpret(Disease.SICKLE_CELL, probability)
            // Spread across a few days so the trend chart isn't a single spike.
            val timestamp = Calendar.getInstance().apply {
                timeInMillis = now
                add(Calendar.DAY_OF_YEAR, -(day % 4))
            }.timeInMillis
            day++
            ScreeningRecord(
                timestampMillis = timestamp,
                disease = Disease.SICKLE_CELL.storageKey,
                result = result.status.name.lowercase(),
                confidencePercent = result.confidencePercent,
                referralFlag = result.status != ScreeningInterpreter.Status.NEGATIVE,
            )
        }

        // Ground truth from the known probabilities: 9 confidently positive
        // (>=65% confidence, including pos_21.jpg at 72.56%), 2 confidently
        // negative, 0 borderline -- none of these 11 real images happen to
        // land in the 50-65% band.
        val stats = DashboardAggregator.compute(records, nowMillis = now)
        assertEquals(11, stats.totalAllDiseases)
        val sc = stats.perDisease.first { it.disease == Disease.SICKLE_CELL }
        assertEquals(11, sc.total)
        assertEquals(9, sc.positiveCount)
        assertEquals(2, sc.negativeCount)
        assertEquals(0, sc.borderlineCount)
        assertEquals(9, sc.referralCount) // positive only, since none borderline here
        assertEquals(9, sc.dailyCounts.sumOf { it.count }) // the 9 positives, all within the 7-day window
        // No malaria screenings logged in this simulation -- confirms they stay at zero, not blended in.
        val mal = stats.perDisease.first { it.disease == Disease.MALARIA }
        assertEquals(0, mal.total)

        val csv = CsvExporter.toCsv(records)
        println("---- Simulated CSV export ----")
        println(csv)
        println("---- Dashboard stats (sickle cell) ----")
        println(
            "total=${sc.total} positive=${sc.positiveCount} negative=${sc.negativeCount} " +
                "borderline=${sc.borderlineCount} referrals=${sc.referralCount} " +
                "positive%=${sc.positivePercent} negative%=${sc.negativePercent}"
        )

        val lines = csv.trim().split("\n")
        assertEquals(1 + 11, lines.size) // header + 11 rows
        assertEquals("timestamp,disease,result,confidence_percent,referral_flag", lines[0])
        // 9 positive rows + 2 negative rows, referral_flag "yes" exactly on the positive ones
        assertEquals(9, lines.drop(1).count { it.contains(",positive,") && it.endsWith(",yes") })
        assertEquals(2, lines.drop(1).count { it.contains(",negative,") && it.endsWith(",no") })
        assertEquals(11, lines.drop(1).count { it.contains(",sickle_cell,") })
    }
}
