package com.sicklescan.app

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.sicklescan.app.data.CsvExporter
import com.sicklescan.app.data.DashboardAggregator
import com.sicklescan.app.data.DiseaseStats
import com.sicklescan.app.data.ScreeningRepository
import com.sicklescan.app.databinding.FragmentDashboardBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Locale

/**
 * Aggregate stats + trend + CSV export over the local screening log. This
 * is a single device's own activity, not a synced multi-worker view --
 * that note is shown directly in the layout (dashboard_local_note), not
 * buried in a dialog.
 */
class DashboardFragment : Fragment() {

    private var _binding: FragmentDashboardBinding? = null
    private val binding get() = _binding!!

    private lateinit var repository: ScreeningRepository

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?,
    ): View {
        _binding = FragmentDashboardBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        repository = ScreeningRepository(requireContext())
        binding.btnExportCsv.setOnClickListener { onExportClicked() }
    }

    override fun onResume() {
        super.onResume()
        // Refresh every time this tab is shown, so a screening just logged
        // on the Screen tab is reflected immediately.
        refresh()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    private fun refresh() {
        viewLifecycleOwner.lifecycleScope.launch {
            val records = withContext(Dispatchers.IO) { repository.getAllRecords() }
            val sessions = withContext(Dispatchers.IO) { repository.getAllSessions() }
            val stats = DashboardAggregator.compute(sessions, records)

            if (_binding == null) return@launch // view may be gone by the time this resumes

            if (stats.sessionCount == 0 && stats.overriddenSessionCount == 0 && stats.inconclusiveChecks == 0) {
                binding.emptyText.visibility = View.VISIBLE
                binding.totalText.visibility = View.GONE
                binding.sessionsDetailText.visibility = View.GONE
                binding.statsContainer.visibility = View.GONE
                return@launch
            }

            binding.emptyText.visibility = View.GONE
            binding.totalText.visibility = View.VISIBLE
            // Headline is SESSIONS (photos), not summed condition checks, so a photo screened
            // for two conditions isn't counted twice. The per-condition numbers below are the
            // secondary breakdown, reconciled explicitly.
            binding.totalText.text = getString(R.string.dashboard_sessions_format, stats.sessionCount)
            val details = mutableListOf<String>()
            if (stats.bothConditionsSessions > 0) {
                details += getString(R.string.dashboard_sessions_both_format, stats.bothConditionsSessions)
            }
            if (stats.overriddenSessionCount > 0) {
                details += getString(R.string.dashboard_sessions_overridden_format, stats.overriddenSessionCount)
            }
            if (stats.inconclusiveChecks > 0) {
                details += getString(R.string.dashboard_inconclusive_format, stats.inconclusiveChecks)
            }
            binding.sessionsDetailText.text = details.joinToString("\n")
            binding.sessionsDetailText.visibility = if (details.isEmpty()) View.GONE else View.VISIBLE

            if (stats.sessionCount == 0) {
                binding.statsContainer.visibility = View.GONE
                return@launch
            }
            binding.statsContainer.visibility = View.VISIBLE
            binding.reconcileText.text = getString(
                R.string.dashboard_reconcile_format,
                stats.sessionCount, stats.conditionChecks, stats.bothConditionsSessions,
            )

            val sc = stats.perDisease.first { it.disease == Disease.SICKLE_CELL }
            bindDiseaseSection(sc, binding.scEmptyText, binding.scStatsGroup, binding.scBreakdownText, binding.scReferralText, binding.scBarChart)

            val mal = stats.perDisease.first { it.disease == Disease.MALARIA }
            bindDiseaseSection(mal, binding.malEmptyText, binding.malStatsGroup, binding.malBreakdownText, binding.malReferralText, binding.malBarChart)
        }
    }

    private fun bindDiseaseSection(
        stats: DiseaseStats,
        emptyText: android.widget.TextView,
        statsGroup: View,
        breakdownText: android.widget.TextView,
        referralText: android.widget.TextView,
        barChart: BarChartView,
    ) {
        if (stats.total == 0) {
            emptyText.visibility = View.VISIBLE
            statsGroup.visibility = View.GONE
            return
        }
        emptyText.visibility = View.GONE
        statsGroup.visibility = View.VISIBLE

        breakdownText.text = getString(
            R.string.dashboard_breakdown_format,
            stats.total,
            stats.positivePercent,
            stats.negativePercent,
            stats.borderlinePercent,
        )
        referralText.text = getString(R.string.dashboard_referral_format, stats.referralCount)
        barChart.setData(stats.dailyCounts)
    }

    private fun onExportClicked() {
        viewLifecycleOwner.lifecycleScope.launch {
            val records = withContext(Dispatchers.IO) { repository.getAllRecords() }
            val sessions = withContext(Dispatchers.IO) { repository.getAllSessions() }
            if (records.isEmpty()) {
                Toast.makeText(requireContext(), getString(R.string.export_empty), Toast.LENGTH_SHORT).show()
                return@launch
            }

            try {
                val uri = withContext(Dispatchers.IO) { writeCsvFile(CsvExporter.toCsv(sessions, records)) }
                shareCsv(uri)
            } catch (e: IOException) {
                Log.e(TAG, "CSV export failed", e)
                Toast.makeText(requireContext(), getString(R.string.export_failed), Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun writeCsvFile(csv: String): android.net.Uri {
        val exportsDir = File(requireContext().cacheDir, "exports").apply { mkdirs() }
        val fileName = "sicklescan_log_${FILENAME_TIMESTAMP.format(System.currentTimeMillis())}.csv"
        val file = File(exportsDir, fileName)
        file.writeText(csv)
        return FileProvider.getUriForFile(requireContext(), "${requireContext().packageName}.fileprovider", file)
    }

    private fun shareCsv(uri: android.net.Uri) {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/csv"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(intent, getString(R.string.export_chooser_title)))
    }

    companion object {
        private const val TAG = "DashboardFragment"
        private val FILENAME_TIMESTAMP = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
    }
}
