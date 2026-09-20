package com.sicklescan.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.StringRes
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.sicklescan.app.data.CaptureSession
import com.sicklescan.app.data.LoggedResult
import com.sicklescan.app.data.ScreeningRepository
import com.sicklescan.app.databinding.FragmentScreenBinding
import com.sicklescan.app.databinding.ItemResultCardBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.IOException

/**
 * Capture-first flow (Phase 8): capture/select ONE photo -> guardrail check (once)
 * -> choose which condition(s) to screen for (Sickle Cell, Malaria, or both) ->
 * run the selected model(s) sequentially against that SAME photo -> one result
 * card per condition. Each run is logged as one [CaptureSession] with 1-2
 * screening records. There is no auto-detection of which disease an image is for.
 */
class ScreenFragment : Fragment() {

    private enum class GuardrailState { NONE, ACCEPTED, OVERRIDDEN }

    private var _binding: FragmentScreenBinding? = null
    private val binding get() = _binding!!

    // One classifier per disease, loaded lazily on first use so a load
    // failure for one model (e.g. a corrupt asset) doesn't block the other.
    private val classifiers = mutableMapOf<Disease, ImageClassifier>()
    private val classifierLoadFailed = mutableSetOf<Disease>()

    // Phase 7 guardrail: runs once per photo, before any disease model.
    private var guardrail: ImageClassifier? = null
    private var guardrailLoadFailed = false

    private lateinit var repository: ScreeningRepository
    private var selectedBitmap: Bitmap? = null
    private var guardrailState = GuardrailState.NONE

    // Row id of the guardrail rejection currently on screen, linked to the session if the user continues anyway.
    private var pendingRejectionId: Long? = null

    // --- Activity result launchers ---

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) launchCamera() else showError(getString(R.string.error_camera_permission))
    }

    private val storagePermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) launchGalleryPicker() else showError(getString(R.string.error_storage_permission))
    }

    private val cameraCaptureLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == android.app.Activity.RESULT_OK) {
            result.data?.data?.let { loadImageFromUri(it) }
        }
    }

    // Android Photo Picker (API 33+ native, backed by Play services on older
    // devices too) — requires no storage permission at all.
    private val galleryPickerLauncher = registerForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri: Uri? ->
        if (uri != null) loadImageFromUri(uri)
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?,
    ): View {
        _binding = FragmentScreenBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        repository = ScreeningRepository(requireContext())

        binding.btnTakePhoto.setOnClickListener { onTakePhotoClicked() }
        binding.btnChooseGallery.setOnClickListener { onChooseGalleryClicked() }
        binding.btnAnalyze.setOnClickListener { onAnalyzeClicked() }
        binding.btnGuardrailRetake.setOnClickListener { onTakePhotoClicked() }
        binding.btnGuardrailAnyway.setOnClickListener { onContinueAnywayClicked() }

        val updateAnalyze = { updateAnalyzeEnabled() }
        binding.checkSickleCell.setOnCheckedChangeListener { _, _ -> updateAnalyze() }
        binding.checkMalaria.setOnCheckedChangeListener { _, _ -> updateAnalyze() }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    override fun onDestroy() {
        super.onDestroy()
        classifiers.values.forEach { it.close() }
        guardrail?.close()
    }

    /** Lazily loads (and caches) the classifier for [disease]. Returns null if that
     * specific model fails to load -- never crashes and never blocks the other model. */
    private fun getClassifier(disease: Disease): ImageClassifier? {
        classifiers[disease]?.let { return it }
        if (disease in classifierLoadFailed) return null
        return try {
            ImageClassifier(requireContext(), disease).also { classifiers[disease] = it }
        } catch (e: ImageClassifier.ClassifierException) {
            Log.e(TAG, "Model load failed for $disease", e)
            classifierLoadFailed += disease
            null
        }
    }

    private fun getGuardrail(): ImageClassifier? {
        guardrail?.let { return it }
        if (guardrailLoadFailed) return null
        return try {
            ImageClassifier(requireContext(), GUARDRAIL_MODEL_ASSET, GUARDRAIL_LABELS_ASSET).also { guardrail = it }
        } catch (e: ImageClassifier.ClassifierException) {
            Log.e(TAG, "Guardrail model load failed", e)
            guardrailLoadFailed = true
            null
        }
    }

    // --- Button handlers ---

    private fun onTakePhotoClicked() {
        when {
            ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED -> launchCamera()
            else -> cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun onChooseGalleryClicked() {
        // The Photo Picker contract needs no runtime permission on any supported API level (24+);
        // the legacy permission path is only a defensive fallback where the picker is unavailable.
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.S_V2 &&
            ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.READ_EXTERNAL_STORAGE) !=
            PackageManager.PERMISSION_GRANTED &&
            !isPhotoPickerAvailable()
        ) {
            storagePermissionLauncher.launch(Manifest.permission.READ_EXTERNAL_STORAGE)
        } else {
            launchGalleryPicker()
        }
    }

    /** New photo -> reset the flow and run the guardrail check ONCE for it. */
    private fun runGuardrail(bitmap: Bitmap) {
        val guard = getGuardrail()
        if (guard == null) {
            showError(getString(R.string.error_model_load))
            return
        }
        setBusy(true, R.string.checking_image)
        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val smearProbability = withContext(Dispatchers.Default) { guard.classify(bitmap) }
                if (GuardrailInterpreter.looksLikeSmear(smearProbability)) {
                    guardrailState = GuardrailState.ACCEPTED
                    showSelection(overridden = false)
                } else {
                    // Rejected BEFORE any disease model runs. Logged in its own table.
                    pendingRejectionId = withContext(Dispatchers.IO) { repository.logGuardrailRejection(smearProbability) }
                    binding.guardrailContainer.visibility = View.VISIBLE
                }
            } catch (e: ImageClassifier.ClassifierException) {
                Log.e(TAG, "Guardrail inference failed", e)
                showError(getString(R.string.error_inference))
            } finally {
                setBusy(false)
            }
        }
    }

    /** Soft warning: the user chose to continue past the guardrail. Nothing is logged yet --
     * the session (marked overridden) is written when they tap Analyze. */
    private fun onContinueAnywayClicked() {
        guardrailState = GuardrailState.OVERRIDDEN
        binding.guardrailContainer.visibility = View.GONE
        showSelection(overridden = true)
    }

    private fun showSelection(overridden: Boolean) {
        binding.overrideNote.visibility = if (overridden) View.VISIBLE else View.GONE
        binding.selectionContainer.visibility = View.VISIBLE
        updateAnalyzeEnabled()
    }

    private fun selectedDiseases(): List<Disease> = buildList {
        if (binding.checkSickleCell.isChecked) add(Disease.SICKLE_CELL)
        if (binding.checkMalaria.isChecked) add(Disease.MALARIA)
    }

    /** Runs the selected model(s) sequentially against the SAME bitmap, then logs one session. */
    private fun onAnalyzeClicked() {
        val bitmap = selectedBitmap
        val diseases = selectedDiseases()
        if (bitmap == null) {
            showError(getString(R.string.error_no_image))
            return
        }
        if (diseases.isEmpty() || guardrailState == GuardrailState.NONE) return

        val loaded = diseases.map { it to getClassifier(it) }
        if (loaded.any { it.second == null }) {
            showError(getString(R.string.error_model_load))
            return
        }

        val overridden = guardrailState == GuardrailState.OVERRIDDEN
        binding.resultsContainer.removeAllViews()
        binding.crossDomainWarning.visibility = View.GONE
        binding.errorText.visibility = View.GONE
        setBusy(true, R.string.analyzing)

        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val started = System.currentTimeMillis()
                // Sequential, one image: no re-capture between models.
                val outcomes = withContext(Dispatchers.Default) {
                    loaded.map { (disease, classifier) ->
                        disease to ScreeningInterpreter.interpret(disease, classifier!!.classify(bitmap))
                    }
                }
                val elapsed = System.currentTimeMillis() - started
                if (elapsed < MIN_LOADING_MS) delay(MIN_LOADING_MS - elapsed)

                // Only after every selected model succeeded: one session, 1-2 records, atomically.
                withContext(Dispatchers.IO) {
                    repository.logSession(
                        guardrailResult = if (overridden) CaptureSession.GUARDRAIL_OVERRIDDEN else CaptureSession.GUARDRAIL_ACCEPTED,
                        results = outcomes.map { (disease, result) ->
                            LoggedResult(
                                disease = disease,
                                result = result.status.name.lowercase(),
                                confidencePercent = result.confidencePercent,
                                referralFlag = result.status != ScreeningInterpreter.Status.NEGATIVE,
                            )
                        },
                        rejectionEventId = if (overridden) pendingRejectionId else null,
                    )
                }
                pendingRejectionId = null

                outcomes.forEach { (disease, result) -> addResultCard(disease, result, overridden) }
                binding.disclaimerText.visibility = View.GONE // each card carries its own disclaimer
                // Shown on EVERY result screen that displays two conditions together (never blocks Both).
                binding.crossDomainWarning.visibility =
                    if (ResultsPresentation.showCrossDomainWarning(outcomes.map { it.first })) View.VISIBLE else View.GONE
            } catch (e: ImageClassifier.ClassifierException) {
                Log.e(TAG, "Inference failed", e)
                showError(getString(R.string.error_inference))
            } finally {
                setBusy(false)
            }
        }
    }

    // --- Helpers ---

    private fun launchCamera() {
        cameraCaptureLauncher.launch(Intent(requireContext(), CameraActivity::class.java))
    }

    private fun launchGalleryPicker() {
        galleryPickerLauncher.launch(
            PickVisualMediaRequest.Builder()
                .setMediaType(ActivityResultContracts.PickVisualMedia.ImageOnly)
                .build()
        )
    }

    private fun isPhotoPickerAvailable(): Boolean {
        return ActivityResultContracts.PickVisualMedia.isPhotoPickerAvailable(requireContext())
    }

    private fun loadImageFromUri(uri: Uri) {
        try {
            requireContext().contentResolver.openInputStream(uri).use { stream ->
                val bitmap = stream?.let { BitmapFactory.decodeStream(it) }
                if (bitmap == null) {
                    showError(getString(R.string.error_corrupt_image))
                    return
                }
                resetFlow()
                selectedBitmap = bitmap
                binding.imagePreview.setImageBitmap(bitmap)
                binding.noImageText.visibility = View.GONE
                runGuardrail(bitmap)
            }
        } catch (e: IOException) {
            Log.e(TAG, "Failed to read image", e)
            showError(getString(R.string.error_corrupt_image))
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "Image too large to decode", e)
            showError(getString(R.string.error_corrupt_image))
        }
    }

    /** Clears everything tied to the previous photo. */
    private fun resetFlow() {
        guardrailState = GuardrailState.NONE
        pendingRejectionId = null
        binding.guardrailContainer.visibility = View.GONE
        binding.selectionContainer.visibility = View.GONE
        binding.errorText.visibility = View.GONE
        binding.resultsContainer.removeAllViews()
        binding.crossDomainWarning.visibility = View.GONE
        binding.disclaimerText.visibility = View.VISIBLE
        binding.checkSickleCell.isChecked = false
        binding.checkMalaria.isChecked = false
    }

    private fun updateAnalyzeEnabled() {
        binding.btnAnalyze.isEnabled = selectedBitmap != null && selectedDiseases().isNotEmpty()
    }

    /** One card per condition -- own confidence, referral status, disclaimer; never merged. */
    private fun addResultCard(disease: Disease, result: ScreeningInterpreter.ScreeningResult, overridden: Boolean) {
        val card = ItemResultCardBinding.inflate(layoutInflater, binding.resultsContainer, false)

        val (iconRes, colorRes, bgRes) = when (result.status) {
            ScreeningInterpreter.Status.POSITIVE ->
                Triple(R.drawable.ic_status_positive, R.color.status_positive, R.color.status_positive_bg)
            ScreeningInterpreter.Status.BORDERLINE ->
                Triple(R.drawable.ic_status_borderline, R.color.status_borderline, R.color.status_borderline_bg)
            ScreeningInterpreter.Status.NEGATIVE ->
                Triple(R.drawable.ic_status_negative, R.color.status_negative, R.color.status_negative_bg)
        }
        val color = ContextCompat.getColor(requireContext(), colorRes)

        card.resultCard.setCardBackgroundColor(ContextCompat.getColor(requireContext(), bgRes))
        card.diseaseTitle.text = disease.displayName
        card.resultIcon.setImageResource(iconRes)
        card.resultIcon.imageTintList = android.content.res.ColorStateList.valueOf(color)

        // Result wording is always a screening indication, never a diagnosis.
        card.resultLabel.text = getString(R.string.result_label_format, result.resultLabel)
        card.resultLabel.setTextColor(color)
        card.resultConfidence.text = getString(R.string.confidence_format, result.confidencePercent)
        card.referralMessage.text = result.referralMessage
        card.referralMessage.setTextColor(color)
        card.overrideCaveat.visibility = if (overridden) View.VISIBLE else View.GONE

        binding.resultsContainer.addView(card.root)
    }

    private fun showError(message: String) {
        binding.guardrailContainer.visibility = View.GONE
        binding.selectionContainer.visibility = View.GONE
        binding.resultsContainer.removeAllViews()
        binding.crossDomainWarning.visibility = View.GONE
        binding.errorText.visibility = View.VISIBLE
        binding.errorText.text = message
    }

    private fun setBusy(busy: Boolean, @StringRes message: Int = R.string.analyzing) {
        binding.loadingContainer.visibility = if (busy) View.VISIBLE else View.GONE
        binding.loadingText.setText(message)
        binding.btnTakePhoto.isEnabled = !busy
        binding.btnChooseGallery.isEnabled = !busy
        binding.checkSickleCell.isEnabled = !busy
        binding.checkMalaria.isEnabled = !busy
        binding.btnAnalyze.isEnabled = !busy && selectedBitmap != null && selectedDiseases().isNotEmpty()
    }

    companion object {
        private const val TAG = "ScreenFragment"
        private const val GUARDRAIL_MODEL_ASSET = "guardrail_model.tflite"
        private const val GUARDRAIL_LABELS_ASSET = "guardrail_labels.txt"
        private const val MIN_LOADING_MS = 400L
    }
}
