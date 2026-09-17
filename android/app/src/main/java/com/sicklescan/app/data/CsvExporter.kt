package com.sicklescan.app.data

import java.text.SimpleDateFormat
import java.util.Locale
import java.util.TimeZone

/**
 * Builds the CSV export as a plain string -- pure and file-I/O free, so it's
 * a real, runnable local unit test (see CsvExporterTest). Writing the string
 * to a file and sharing it is separate (MainActivity/DashboardFragment),
 * where it belongs given it needs a Context.
 *
 * This CSV *is* the integration point with a district/PHC coordinator's
 * system in the current scope -- there is no server to upload to, and this
 * app doesn't pretend otherwise.
 */
object CsvExporter {

    private val TIMESTAMP_FORMAT = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US).apply {
        timeZone = TimeZone.getDefault()
    }

    fun toCsv(records: List<ScreeningRecord>): String {
        val builder = StringBuilder()
        builder.append("timestamp,result,confidence_percent,referral_flag\n")
        // Exported oldest-first, since that's the natural reading order for
        // a log a coordinator would scan top to bottom.
        records.sortedBy { it.timestampMillis }.forEach { record ->
            builder.append(escapeCsv(TIMESTAMP_FORMAT.format(record.timestampMillis)))
            builder.append(',')
            builder.append(escapeCsv(record.result))
            builder.append(',')
            builder.append(String.format(Locale.US, "%.1f", record.confidencePercent))
            builder.append(',')
            builder.append(if (record.referralFlag) "yes" else "no")
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
