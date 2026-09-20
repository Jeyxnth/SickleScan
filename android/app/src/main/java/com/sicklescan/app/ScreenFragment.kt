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
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.sicklescan.app.data.ScreeningRepository
import com.sicklescan.app.databinding.FragmentScreenBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.IOException

/**
 * The main capture -> select disease -> classify -> result flow. Two
 * models are bundled (see [Disease]); the user picks which one to run
 * before analyzing -- there's no attempt to auto-detect which disease an
 * image is for for, that's a different, harder problem and out of scope.
 */
class ScreenFragment : Fragment() {

    private var _binding: FragmentScreenBinding? = null
    private val binding get() = _binding!!

    // One classifier per disease, loaded lazily on first use so a load
    // failure for one model (e.g. a corrupt asset) doesn't block the other.
    private val classifiers = mutableMapOf<Disease, ImageClassifier>()
    private val classifierLoadFailed = mutableSetOf<Disease>()

    // Phase 7 guardrail: runs before any disease model. Lazily loaded like the others.
    private var guardrail: ImageClassifier? = null
    private var guardrailLoadFailed = false

    // Row id of the guardrail rejection currently on screen, so "Analyze anyway" can mark it overridden.
    private var pendingRejectionId: Long? = null

    private lateinit var repository: ScreeningRepository
    private var selectedDisease: Disease = Disease.SICKLE_CELL
    private var selectedBitmap: Bitmap? = null

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

        binding.diseaseToggle.check(R.id.btnDiseaseSickleCell)
        binding.diseaseToggle.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (!isChecked) return@addOnButtonCheckedListener
            selectedDisease = when (checkedId) {
                R.id.btnDiseaseMalaria -> Disease.MALARIA
                else -> Disease.SICKLE_CELL
            }
            // A result/referral for one disease doesn't apply to another;
            // clear it rather than leave a stale verdict on screen.
            clearResult()
            updateAnalyzeEnabled()
        }

        binding.btnTakePhoto.setOnClickListener { onTakePhotoClicked() }
        binding.btnChooseGallery.setOnClickListener { onChooseGalleryClicked() }
        binding.btnAnalyze.setOnClickListener { onAnalyzeClicked() }
        binding.btnGuardrailRetake.setOnClickListener { onTakePhotoClicked() }
        binding.btnGuardrailAnyway.setOnClickListener { onAnalyzeAnywayClicked() }
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

    /** Lazily loads (and caches) the classifier for [disease]. Returns null,
     * having shown an error, if that specific model fails to load -- this
     * never crashes and never blocks the other disease's model. */
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
        clearResult()
        when {
            ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED -> launchCamera()
            else -> cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun onChooseGalleryClicked() {
        clearResult()
        // The Photo Picker contract needs no runtime permission on any
        // supported API level (24+); it launches a system UI in a separate
        // process. Kept the legacy permission path only as a defensive
        // fallback for the (very rare) device where the picker is unavailable.
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

    /** Capture -> preprocess -> GUARDRAIL check -> (if smear-like) disease classifier -> result,
     * or (if not smear-like) -> warning with Retake / Analyze anyway. */
    private fun onAnalyzeClicked() {
        val bitmap = selectedBitmap
        val disease = selectedDisease

        if (bitmap == null) {
            showError(getString(R.string.error_no_image))
            return
        }
        val guard = getGuardrail()
        if (guard == null || getClassifier(disease) == null) {
            showError(getString(R.string.error_model_load))
            return
        }

        clearResult()
        setBusy(true)

        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val smearProbability = withContext(Dispatchers.Default) { guard.classify(bitmap) }
                if (!GuardrailInterpreter.looksLikeSmear(smearProbability)) {
                    // Rejected BEFORE the disease model runs. Logged in its own table.
                    pendingRejectionId = withContext(Dispatchers.IO) {
                        repository.logGuardrailRejection(disease, smearProbability)
                    }
                    showGuardrailWarning()
                    return@launch
                }
                runDiseaseAnalysis(bitmap, disease, overridden = false)
            } catch (e: ImageClassifier.ClassifierException) {
                Log.e(TAG, "Inference failed", e)
                showError(getString(R.string.error_inference))
            } finally {
                setBusy(false)
            }
        }
    }

    /** Soft-warning path: the user chose to analyze despite the guardrail. The result is shown
     * with a persistent caveat and is NOT logged as a disease screening (so it can't skew stats);
     * the rejection row is marked overridden instead. */
    private fun onAnalyzeAnywayClicked() {
        val bitmap = selectedBitmap ?: return
        val disease = selectedDisease
        val rejectionId = pendingRejectionId

        clearResult()
        setBusy(true)
        viewLifecycleOwner.lifecycleScope.launch {
            try {
                if (rejectionId != null) withContext(Dispatchers.IO) { repository.markGuardrailOverridden(rejectionId) }
                pendingRejectionId = null
                runDiseaseAnalysis(bitmap, disease, overridden = true)
            } catch (e: ImageClassifier.ClassifierException) {
                Log.e(TAG, "Inference failed", e)
                showError(getString(R.string.error_inference))
            } finally {
                setBusy(false)
            }
        }
    }

    private suspend fun runDiseaseAnalysis(bitmap: Bitmap, disease: Disease, overridden: Boolean) {
        val classifier = getClassifier(disease)
        if (classifier == null) {
            showError(getString(R.string.error_model_load))
            return
        }
        // Inference runs off the main thread so the loading state is always visibly shown.
        val (probability, elapsedMs) = withContext(Dispatchers.Default) {
            val start = System.currentTimeMillis()
            val p = classifier.classify(bitmap)
            p to (System.currentTimeMillis() - start)
        }
        val minVisibleMs = 400L
        if (elapsedMs < minVisibleMs) delay(minVisibleMs - elapsedMs)

        val result = ScreeningInterpreter.interpret(disease, probability)
        showResult(result)
        binding.overrideCaveat.visibility = if (overridden) View.VISIBLE else View.GONE

        if (!overridden) {
            // Log every guardrail-passed screening, tagged with disease. No image is ever stored.
            withContext(Dispatchers.IO) {
                repository.logScreening(
                    disease = disease,
                    result = result.status.name.lowercase(),
                    confidencePercent = result.confidencePercent,
                    referralFlag = result.status != ScreeningInterpreter.Status.NEGATIVE,
                )
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
                selectedBitmap = bitmap
                binding.imagePreview.setImageBitmap(bitmap)
                binding.noImageText.visibility = View.GONE
                updateAnalyzeEnabled()
            }
        } catch (e: IOException) {
            Log.e(TAG, "Failed to read image", e)
            showError(getString(R.string.error_corrupt_image))
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "Image too large to decode", e)
            showError(getString(R.string.error_corrupt_image))
        }
    }

    private fun updateAnalyzeEnabled() {
        binding.btnAnalyze.isEnabled = selectedBitmap != null
    }

    private fun showResult(result: ScreeningInterpreter.ScreeningResult) {
        binding.errorText.visibility = View.GONE
        binding.resultContainer.visibility = View.VISIBLE

        val (iconRes, colorRes, bgRes) = when (result.status) {
            ScreeningInterpreter.Status.POSITIVE ->
                Triple(R.drawable.ic_status_positive, R.color.status_positive, R.color.status_positive_bg)
            ScreeningInterpreter.Status.BORDERLINE ->
                Triple(R.drawable.ic_status_borderline, R.color.status_borderline, R.color.status_borderline_bg)
            ScreeningInterpreter.Status.NEGATIVE ->
                Triple(R.drawable.ic_status_negative, R.color.status_negative, R.color.status_negative_bg)
        }
        val color = ContextCompat.getColor(requireContext(), colorRes)

        binding.resultContainer.setCardBackgroundColor(ContextCompat.getColor(requireContext(), bgRes))
        binding.resultIcon.setImageResource(iconRes)
        binding.resultIcon.imageTintList = android.content.res.ColorStateList.valueOf(color)

        // Result wording is always a screening indication, never a diagnosis.
        binding.resultLabel.text = getString(R.string.result_label_format, result.resultLabel)
        binding.resultLabel.setTextColor(color)

        binding.resultConfidence.text = getString(R.string.confidence_format, result.confidencePercent)

        binding.referralMessage.text = result.referralMessage
        binding.referralMessage.setTextColor(color)
    }

    private fun showError(message: String) {
        binding.guardrailContainer.visibility = View.GONE
        binding.resultContainer.visibility = View.GONE
        binding.errorText.visibility = View.VISIBLE
        binding.errorText.text = message
    }

    private fun showGuardrailWarning() {
        binding.resultContainer.visibility = View.GONE
        binding.errorText.visibility = View.GONE
        binding.guardrailContainer.visibility = View.VISIBLE
    }

    private fun clearResult() {
        binding.guardrailContainer.visibility = View.GONE
        binding.resultContainer.visibility = View.GONE
        binding.errorText.visibility = View.GONE
    }

    private fun setBusy(busy: Boolean) {
        binding.loadingContainer.visibility = if (busy) View.VISIBLE else View.GONE
        binding.btnAnalyze.isEnabled = !busy && selectedBitmap != null
        binding.btnTakePhoto.isEnabled = !busy
        binding.btnChooseGallery.isEnabled = !busy
        binding.diseaseToggle.isEnabled = !busy
    }

    companion object {
        private const val TAG = "ScreenFragment"
        private const val GUARDRAIL_MODEL_ASSET = "guardrail_model.tflite"
        private const val GUARDRAIL_LABELS_ASSET = "guardrail_labels.txt"
    }
}
