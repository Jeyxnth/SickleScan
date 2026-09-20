package com.sicklescan.app

/**
 * Turns the guardrail model's raw sigmoid output into an accept/reject
 * decision. The guardrail is a binary "is this a blood smear microscopy
 * image?" classifier that runs BEFORE any disease model; its output is
 * P(smear) (labels: not_smear, smear). Pure Kotlin so it is unit-testable
 * locally, like [ScreeningInterpreter].
 *
 * What it does and does not mean: it detects whether an image resembles the
 * smear-like photos it was trained on (sickle cell fields + malaria cell
 * crops) versus everyday photos, faces, screenshots, documents, textures and
 * blank/blurred/finger-covered captures. It was NOT trained against other
 * microscopy or histology images, so it is not a "blood smear vs other
 * microscopy" detector -- see model_output/guardrail/guardrail_results.md.
 *
 * Threshold: 0.5 is the default rather than a tuned value. On validation data
 * every threshold from 0.02 to 0.98 made identical errors (the model's outputs
 * are almost entirely 0 or 1), so there was nothing to tune against; the
 * threshold is reported honestly as un-validated in the results file.
 */
object GuardrailInterpreter {

    /** P(smear) below this => the image doesn't look like a blood smear. */
    const val REJECT_BELOW = 0.5f

    fun looksLikeSmear(smearProbability: Float): Boolean = smearProbability >= REJECT_BELOW
}
