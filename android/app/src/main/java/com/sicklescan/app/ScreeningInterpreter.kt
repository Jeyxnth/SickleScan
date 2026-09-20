package com.sicklescan.app

/**
 * Turns a model's raw sigmoid output into what the result screen shows:
 * a plain-language result, a confidence percentage, a status for visual
 * styling, and a referral message. Pure Kotlin / no Android or TFLite
 * dependency, so this is the one piece of this logic that can run as a
 * real local unit test on any machine (see ScreeningInterpreterTest).
 *
 * No severity score: neither classifier was trained or evaluated on
 * anything but whole-field images (see the results.md files under
 * model_output), with no tile/region-level ground truth anywhere in
 * either dataset. A
 * sliding-window "% of tiles positive" severity estimate was considered
 * for the sickle cell model and rejected -- it would report an
 * unvalidated number as if it were a clinical severity metric. Model
 * confidence is shown instead, labeled plainly as confidence, not
 * severity, per that decision -- same reasoning applies to malaria.
 *
 * Borderline threshold is per-[Disease], not one constant reused
 * everywhere: a confidence distribution that means "uncertain" for one
 * model's test set doesn't necessarily mean the same for another's, since
 * different datasets/classifiers push predictions toward 0/1 differently.
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
    //
    // Sickle cell: 65% was chosen and validated against that model's own
    // test-set confidence distribution in Phase 3.
    //
    // Malaria: also 65%, but independently validated against the malaria
    // model's own test-set confidence distribution (Phase 5), not assumed
    // to transfer just because it worked for sickle cell. Binned test-set
    // accuracy by confidence: below 65% confidence, accuracy is 58.4% (113
    // of 4,134 test images, 2.7%) -- barely better than chance, exactly
    // what "uncertain, refer" should mean. At/above 65%, accuracy jumps to
    // 97.4%. The two models landing on the same number is a coincidence of
    // the data, not an assumption -- see malaria_results.md.
    private val BORDERLINE_CEILING: Map<Disease, Float> = mapOf(
        Disease.SICKLE_CELL to 65f,
        Disease.MALARIA to 65f,
    )

    fun interpret(disease: Disease, positiveProbability: Float): ScreeningResult {
        val ceiling = BORDERLINE_CEILING.getValue(disease)
        val isPositive = positiveProbability >= 0.5f
        val confidence = (if (isPositive) positiveProbability else 1f - positiveProbability) * 100f

        val status = when {
            confidence < ceiling -> Status.BORDERLINE
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
