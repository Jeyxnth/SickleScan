package com.sicklescan.app

/**
 * Turns the model's raw sigmoid output into what the result screen shows:
 * a plain-language result, a confidence percentage, a status for visual
 * styling, and a referral message. Pure Kotlin / no Android or TFLite
 * dependency, so this is the one piece of Phase 3's logic that can run as
 * a real local unit test on any machine (see ScreeningInterpreterTest).
 *
 * No severity score: this classifier was trained and evaluated only on
 * whole-field images (see /model_output/results.md), with no tile/region
 * -level ground truth anywhere in the dataset. A sliding-window "% of
 * tiles positive" severity estimate was considered for this phase and
 * rejected — it would report an unvalidated number as if it were a
 * clinical severity metric. Model confidence is shown instead, labeled
 * plainly as confidence, not severity, per that decision.
 */
object ScreeningInterpreter {

    enum class Status { POSITIVE, BORDERLINE, NEGATIVE }

    data class ScreeningResult(
        val resultLabel: String, // "Positive" / "Negative" -- a screening result, never a diagnosis
        val confidencePercent: Float,
        val status: Status,
        val referralMessage: String,
    )

    // Below this confidence (in whichever class the model favored), the
    // model isn't confident enough either way to trust its own verdict --
    // treated as borderline regardless of which side of 50% it landed on.
    private const val BORDERLINE_CEILING = 65f

    fun interpret(positiveProbability: Float): ScreeningResult {
        val isPositive = positiveProbability >= 0.5f
        val confidence = (if (isPositive) positiveProbability else 1f - positiveProbability) * 100f

        val status = when {
            confidence < BORDERLINE_CEILING -> Status.BORDERLINE
            isPositive -> Status.POSITIVE
            else -> Status.NEGATIVE
        }

        val resultLabel = if (isPositive) "Positive" else "Negative"

        val referralMessage = when (status) {
            Status.POSITIVE -> "Refer for lab confirmation"
            Status.BORDERLINE -> "Uncertain result — refer for lab confirmation"
            Status.NEGATIVE -> "Low risk — routine monitoring"
        }

        return ScreeningResult(
            resultLabel = resultLabel,
            confidencePercent = confidence,
            status = status,
            referralMessage = referralMessage,
        )
    }
}
