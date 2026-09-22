package com.sicklescan.app

import kotlin.math.max

/**
 * Decision logic for the wide-field malaria check (Phase 13): CenterNet cell detector -> crop each detected
 * cell -> the BBBC041 malaria classifier -> field-level "any cell >= threshold" rule. Pure Kotlin / no Android
 * or TFLite dependency, so it is a real local unit test (WideFieldMalariaTest).
 *
 * Operating point (Phase 11c, held-out BBBC041 validation fields: 133 infected, 30 negative; baseline classifier;
 * detector score >= 0.3, no box de-duplication -- identical to what this code does):
 *   field rule "any cell >= 0.985"  ->  infected fields caught 120/133 (90.2%), negative fields flagged 5/30 (16.7%) in
 *   sample; 90.2% / 18.3% (95% CI for the latter 8-35%) when the threshold is chosen on training folds only.
 *   Exact in-sample threshold for >= 90%: 0.98892; 0.985 lies on the same plateau.
 * Caveat that matters: those rates are for P. vivax thin smears, one stain protocol, one microscope setup, and only 30
 * negative fields. Classifier scores compress on unfamiliar microscopes (Phase 9b), so this threshold will not transfer
 * unchanged to other setups.
 */
object WideFieldMalaria {

    const val DETECTOR_INPUT_SIZE = 640
    const val DETECTOR_STRIDE = 4
    const val DETECTOR_SCORE_THRESHOLD = 0.3f
    const val MAX_DETECTIONS = 400
    const val FIELD_THRESHOLD = 0.985f

    /**
     * Minimum-cell-count gate: fewer detected cells than this -> Inconclusive, not a verdict. Real BBBC041 thin-smear fields are
     * dense (validation: min 19 cells, 5th percentile 28, median 69) while natural non-smear photos are sparse (median 3), and
     * without this gate 74% of non-smear photos ended in a Negative/Positive verdict (3.3% Positive) once the guardrail was
     * skipped for this mode; at 20 that falls to 8% / 1.1%. On the 163 validation fields it changes no Positive/not-Positive
     * decision (120/133 infected caught, 5/30 negative flagged, unchanged) and turns one 19-cell negative field from
     * Negative into Inconclusive.
     *
     * KNOWN LIMITATION, NOT A SOLVED PROBLEM: 20 was tuned on dense lab-microscope BBBC041 fields. It has NOT been validated
     * on sparser real phone-captured smears, where a genuinely infected but sparse field could incorrectly fall below the gate
     * and return Inconclusive instead of a real result.
     */
    const val MIN_CELLS = 20

    /** Working-resolution cap for the photo (long side, px); bounds memory. BBBC041 fields (1600-1944 px) are unaffected. */
    const val WORKING_LONG_SIDE = 2048

    /** Shown to the user as context (Phase 11c, validation numbers rounded). */
    const val VALIDATED_SENSITIVITY_PERCENT = 90
    const val VALIDATED_FALSE_ALARM_PERCENT = 18

    class Detection(val x0: Float, val y0: Float, val x1: Float, val y1: Float, val score: Float)

    /**
     * Turns the detector's output map into detections in detector-canvas pixels.
     * [map] is NHWC [gridH, gridW, 5] flattened: channel 0 = cell probability (sigmoid already applied in the model),
     * 1-2 = box width/height in grid cells, 3-4 = centre offset inside the cell. A cell is a detection if its probability is
     * >= [threshold] and >= all 8 neighbours (out-of-grid neighbours count as 0), like the Python decode; the strongest
     * [MAX_DETECTIONS] are kept.
     */
    fun decodeDetections(
        map: FloatArray, gridH: Int, gridW: Int,
        threshold: Float = DETECTOR_SCORE_THRESHOLD, maxDetections: Int = MAX_DETECTIONS,
    ): List<Detection> {
        fun heat(y: Int, x: Int): Float = if (y < 0 || x < 0 || y >= gridH || x >= gridW) 0f else map[(y * gridW + x) * 5]
        val found = ArrayList<Detection>()
        for (y in 0 until gridH) {
            for (x in 0 until gridW) {
                val h = heat(y, x)
                if (h < threshold) continue
                var isPeak = true
                loop@ for (dy in -1..1) for (dx in -1..1) {
                    if ((dy != 0 || dx != 0) && heat(y + dy, x + dx) > h) { isPeak = false; break@loop }
                }
                if (!isPeak) continue
                val o = (y * gridW + x) * 5
                val w = max(map[o + 1], 1f) * DETECTOR_STRIDE
                val hh = max(map[o + 2], 1f) * DETECTOR_STRIDE
                val cx = (x + map[o + 3]) * DETECTOR_STRIDE
                val cy = (y + map[o + 4]) * DETECTOR_STRIDE
                found += Detection(cx - w / 2, cy - hh / 2, cx + w / 2, cy + hh / 2, h)
            }
        }
        found.sortByDescending { it.score }
        return if (found.size > maxDetections) found.subList(0, maxDetections).toList() else found
    }

    enum class FieldStatus { POSITIVE, NEGATIVE, TOO_FEW_CELLS }

    class FieldResult(val status: FieldStatus, val cellsAnalysed: Int, val topCellScore: Float)

    /**
     * "Any cell >= [FIELD_THRESHOLD]" -> positive, but only when at least [MIN_CELLS] cells were detected: fewer than that
     * (including none) is TOO_FEW_CELLS -- never "negative" (typical causes: not a wide-field view of a dense thin smear, a
     * lone zoomed-in cell the detector cannot find at that scale, a non-smear photo, or a genuinely sparse field).
     */
    fun interpret(cellScores: FloatArray, threshold: Float = FIELD_THRESHOLD, minCells: Int = MIN_CELLS): FieldResult {
        if (cellScores.size < minCells || cellScores.isEmpty()) {
            return FieldResult(FieldStatus.TOO_FEW_CELLS, cellScores.size, if (cellScores.isEmpty()) 0f else cellScores.max())
        }
        val top = cellScores.max()
        return FieldResult(if (top >= threshold) FieldStatus.POSITIVE else FieldStatus.NEGATIVE, cellScores.size, top)
    }

    /** The field result as the app's standard screening result (Positive / Negative, referral message). TOO_FEW_CELLS has none. */
    fun toScreeningResult(field: FieldResult): ScreeningInterpreter.ScreeningResult? = when (field.status) {
        FieldStatus.POSITIVE -> ScreeningInterpreter.ScreeningResult(
            resultLabel = "Positive",
            confidencePercent = field.topCellScore * 100f,
            status = ScreeningInterpreter.Status.POSITIVE,
            referralMessage = "Refer for lab confirmation",
        )
        FieldStatus.NEGATIVE -> ScreeningInterpreter.ScreeningResult(
            resultLabel = "Negative",
            confidencePercent = field.topCellScore * 100f,
            status = ScreeningInterpreter.Status.NEGATIVE,
            referralMessage = "No cell was flagged — this check misses about 1 in 10 infected fields",
        )
        FieldStatus.TOO_FEW_CELLS -> null
    }
}
