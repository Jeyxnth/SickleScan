package com.sicklescan.app

/**
 * The set of conditions this app can screen for. Each has its own bundled
 * TFLite model + labels file, and (per ScreeningInterpreter) its own
 * confidence-based referral threshold -- one disease's model doesn't
 * necessarily produce the same confidence distribution as another's, so
 * thresholds are not assumed to transfer.
 *
 * Thalassemia is deliberately not here: no lab-confirmed public dataset
 * was found at a usable size/quality (see Phase 5 notes in
 * /model_output), so it's out of scope for this app until one exists.
 */
enum class Disease(
    val displayName: String,
    val modelAsset: String,
    val labelsAsset: String,
    /** Key used for Room records / CSV export -- stable, not shown to users. */
    val storageKey: String,
) {
    SICKLE_CELL(
        displayName = "Sickle Cell",
        modelAsset = "sicklescan_model.tflite",
        labelsAsset = "labels.txt",
        storageKey = "sickle_cell",
    ),
    MALARIA(
        displayName = "Malaria",
        modelAsset = "malaria_model.tflite",
        labelsAsset = "malaria_labels.txt",
        storageKey = "malaria",
    ),
}
