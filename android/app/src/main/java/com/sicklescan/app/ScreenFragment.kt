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
 * The main capture -> classify -> result flow (was MainActivity's whole
 * job before Phase 4 added the Dashboard tab).
 */
class ScreenFragment : Fragment() {

    private var _binding: FragmentScreenBinding? = null
    private val binding get() = _binding!!

    private var classifier: ImageClassifier? = null
    private lateinit var repository: ScreeningRepository
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

        try {
            classifier = ImageClassifier(requireContext())
        } catch (e: ImageClassifier.ClassifierException) {
            Log.e(TAG, "Model load failed", e)
            showError(getString(R.string.error_model_load))
            binding.btnAnalyze.isEnabled = false
        }

        binding.btnTakePhoto.setOnClickListener { onTakePhotoClicked() }
        binding.btnChooseGallery.setOnClickListener { onChooseGalleryClicked() }
        binding.btnAnalyze.setOnClickListener { onAnalyzeClicked() }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier?.close()
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

    private fun onAnalyzeClicked() {
        val bitmap = selectedBitmap
        val activeClassifier = classifier

        if (bitmap == null) {
            showError(getString(R.string.error_no_image))
            return
        }
        if (activeClassifier == null) {
            showError(getString(R.string.error_model_load))
            return
        }

        clearResult()
        setBusy(true)

        // Run inference off the main thread so the loading state is always
        // visibly shown, never a frozen-looking UI, even though inference
        // itself typically takes well under a second on this small model.
        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val (probability, elapsedMs) = withContext(Dispatchers.Default) {
                    val start = System.currentTimeMillis()
                    val p = activeClassifier.classify(bitmap)
                    p to (System.currentTimeMillis() - start)
                }
                // Keep the loading state visible for a minimum stretch so it
                // never reads as a flicker on fast devices.
                val minVisibleMs = 400L
                if (elapsedMs < minVisibleMs) delay(minVisibleMs - elapsedMs)

                val result = ScreeningInterpreter.interpret(probability)
                showResult(result)

                // Log every screening to the local device log. No image is
                // stored -- only the result, confidence, and referral flag.
                withContext(Dispatchers.IO) {
                    repository.logScreening(
                        result = result.status.name.lowercase(),
                        confidencePercent = result.confidencePercent,
                        referralFlag = result.status != ScreeningInterpreter.Status.NEGATIVE,
                    )
                }
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
                selectedBitmap = bitmap
                binding.imagePreview.setImageBitmap(bitmap)
                binding.noImageText.visibility = View.GONE
                binding.btnAnalyze.isEnabled = classifier != null
            }
        } catch (e: IOException) {
            Log.e(TAG, "Failed to read image", e)
            showError(getString(R.string.error_corrupt_image))
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "Image too large to decode", e)
            showError(getString(R.string.error_corrupt_image))
        }
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
        binding.resultContainer.visibility = View.GONE
        binding.errorText.visibility = View.VISIBLE
        binding.errorText.text = message
    }

    private fun clearResult() {
        binding.resultContainer.visibility = View.GONE
        binding.errorText.visibility = View.GONE
    }

    private fun setBusy(busy: Boolean) {
        binding.loadingContainer.visibility = if (busy) View.VISIBLE else View.GONE
        binding.btnAnalyze.isEnabled = !busy && selectedBitmap != null
        binding.btnTakePhoto.isEnabled = !busy
        binding.btnChooseGallery.isEnabled = !busy
    }

    companion object {
        private const val TAG = "ScreenFragment"
    }
}
