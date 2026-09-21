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
    // Malaria (Phase 10, BBBC041-trained classifier): also 65%, validated independently on THIS model's
    // own held-out data rather than carried over from the earlier NIH-trained model. Accuracy below vs
    // at/above 65% confidence: validation (same imaging setup as training) 65% (n=20) vs 98.9%; official
    // BBBC041 test set (different microscope setup, never used for training/selection) 50% (n=52) vs
    // 97.3%. Only ~1% of crops fall below the cutoff. Caveat that the ceiling cannot fix: on the
    // shifted test set infected cells are confidently called uninfected 51% of the time (their scores
    // are compressed toward 0), so a low-risk "Negative" from this model is weaker evidence on images from
    // an unfamiliar microscope/camera than on training-like ones -- see
    // model_output/malaria_bbbc041/malaria_bbbc041_results.md.
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
