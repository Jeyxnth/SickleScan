package com.sicklescan.app.data

import java.text.SimpleDateFormat
import java.util.Locale
import java.util.TimeZone

/**
 * Builds the CSV export as a plain string -- pure and file-I/O free, so it's
 * a real, runnable local unit test (see CsvExporterTest). Writing the string
 * to a file and sharing it is separate (DashboardFragment), where it belongs
 * given it needs a Context.
 *
 * One row per condition screened. Rows from the same photo share a session_id
 * and timestamp, so two conditions run on one photo are visibly linked, not two
 * unrelated entries. image_check is "accepted" or "overridden" (the user continued
 * past the image-check warning; the dashboard excludes those from its stats, but
 * they are exported so a coordinator can see and filter them). input_mode / cells_detected (Phase 13) tell a
 * wide-field malaria row (per-cell scoring, "any cell" rule) from a whole-image one, whose confidence means something else. input_mode / cells_detected (Phase 13) tell a
 * wide-field malaria row (per-cell scoring, "any cell" rule) from a whole-image one, whose confidence means something else.
 *
 * This CSV *is* the integration point with a district/PHC coordinator's
 * system in the current scope -- there is no server to upload to, and
 * this app doesn't pretend otherwise.
 */
object CsvExporter {

    private val TIMESTAMP_FORMAT = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US).apply {
        timeZone = TimeZone.getDefault()
    }

    fun toCsv(sessions: List<CaptureSession>, records: List<ScreeningRecord>): String {
        val sessionById = sessions.associateBy { it.id }
        val builder = StringBuilder()
        builder.append("session_id,timestamp,disease,result,confidence_percent,referral_flag,image_check,input_mode,cells_detected\n")
        // Oldest session first (ids ascend with time), rows of one session adjacent.
        // Records with no matching session can't occur (foreign key) and are skipped.
        records
            .filter { it.sessionId in sessionById }
            .sortedWith(compareBy({ sessionById.getValue(it.sessionId).timestampMillis }, { it.sessionId }, { it.disease }))
            .forEach { record ->
                val session = sessionById.getValue(record.sessionId)
                builder.append(record.sessionId)
                builder.append(',')
                builder.append(escapeCsv(TIMESTAMP_FORMAT.format(session.timestampMillis)))
                builder.append(',')
                builder.append(escapeCsv(record.disease))
                builder.append(',')
                builder.append(escapeCsv(record.result))
                builder.append(',')
                builder.append(String.format(Locale.US, "%.1f", record.confidencePercent))
                builder.append(',')
                builder.append(if (record.referralFlag) "yes" else "no")
                builder.append(',')
                builder.append(escapeCsv(session.guardrailResult))
                builder.append(',')
                // "field" = wide-field malaria (detect cells, classify each; confidence_percent is then the highest
                // single-cell score); "image" = whole-image classification.
                builder.append(if (record.wideField) "field" else "image")
                builder.append(',')
                builder.append(record.cellsDetected)
                builder.append('\n')
            }
        return builder.toString()
    }

    private fun escapeCsv(value: String): String {
        return if (value.contains(',') || value.contains('"') || value.contains('\n')) {
            "\"" + value.replace("\"", "\"\"") + "\""
        } else {
            value
        }
    }
}
