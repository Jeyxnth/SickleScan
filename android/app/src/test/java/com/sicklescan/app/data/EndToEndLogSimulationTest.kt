package com.sicklescan.app.data

import com.sicklescan.app.ScreeningInterpreter
import org.junit.Assert.assertEquals
import org.junit.Test
import java.util.Calendar

/**
 * Simulates "run a handful of screenings" end to end using the real raw
 * probabilities already independently verified (in Python, against the
 * actual sicklescan_model.tflite) for the 11 known sample images from
 * Phases 2-3 -- the closest thing to a real device run available in this
 * environment (see Phase 2/3 notes on why an actual on-device run isn't
 * possible here). Exercises the exact ScreeningInterpreter ->
 * ScreeningRecord -> DashboardAggregator -> CsvExporter pipeline the app
 * uses, and prints the CSV so it can be inspected by hand.
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
    fun `simulated screening log produces correct dashboard stats and CSV`() {
        val now = System.currentTimeMillis()
        var day = 0
        val records = knownProbabilities.map { (_, probability) ->
            val result = ScreeningInterpreter.interpret(probability)
            // Spread across a few days so the trend chart isn't a single spike.
            val timestamp = Calendar.getInstance().apply {
                timeInMillis = now
                add(Calendar.DAY_OF_YEAR, -(day % 4))
            }.timeInMillis
            day++
            ScreeningRecord(
                timestampMillis = timestamp,
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
        assertEquals(11, stats.total)
        assertEquals(9, stats.positiveCount)
        assertEquals(2, stats.negativeCount)
        assertEquals(0, stats.borderlineCount)
        assertEquals(9, stats.referralCount) // positive only, since none borderline here
        assertEquals(11, stats.dailyCounts.sumOf { it.count }) // all within the 7-day window

        val csv = CsvExporter.toCsv(records)
        println("---- Simulated CSV export ----")
        println(csv)
        println("---- Dashboard stats ----")
        println(
            "total=${stats.total} positive=${stats.positiveCount} negative=${stats.negativeCount} " +
                "borderline=${stats.borderlineCount} referrals=${stats.referralCount} " +
                "positive%=${stats.positivePercent} negative%=${stats.negativePercent}"
        )

        val lines = csv.trim().split("\n")
        assertEquals(1 + 11, lines.size) // header + 11 rows
        assertEquals("timestamp,result,confidence_percent,referral_flag", lines[0])
        // 9 positive rows + 2 negative rows, referral_flag "yes" exactly on the positive ones
        assertEquals(9, lines.drop(1).count { it.contains(",positive,") && it.endsWith(",yes") })
        assertEquals(2, lines.drop(1).count { it.contains(",negative,") && it.endsWith(",no") })
    }
}
