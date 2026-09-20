# SickleScan

**AI-powered, fully offline point-of-care blood smear screening for Android.**

SickleScan turns a smartphone camera and a low-cost clip-on microscope lens into a field screening tool for sickle cell disease and malaria — no lab, no internet, and no specialized hardware required. Built for community health workers (ASHA/PHC staff) conducting screening under India's national elimination missions.

Built by **Team PlasmaPav** (Jeyanth S — SRMIST), for **MedVision'26**.

---

## The Problem

Sickle cell disease and malaria remain serious health burdens in India, concentrated heavily in rural and tribal populations — exactly where diagnostic lab infrastructure is often unavailable. Confirming either condition today typically requires a solubility assay, HPLC, or manual microscopic examination by a trained technician: none of which are deployable at the point of care in a rural PHC or a mobile screening camp.

The National Sickle Cell Anaemia Elimination Mission and the National Framework for Malaria Elimination both depend on wide, frequent screening — SickleScan aims to close the gap between where these diseases actually are and where diagnosis is currently possible.

## What It Does

- **Capture** a blood smear photo via the phone camera (CameraX) or select one from the gallery (Android 13+ Photo Picker)
- **Classify** it on-device, fully offline, using a MobileNetV2 model converted to TensorFlow Lite
- **Screen for two conditions independently** — sickle cell disease and malaria, each with its own trained, validated model
- **Get a result in seconds**: classification, confidence score, and a clear referral recommendation — never phrased as a diagnosis
- **Track screening history locally** via an on-device dashboard showing per-disease totals, positive/negative/borderline breakdown, referral counts, and a 7-day trend
- **Export screening logs as CSV** for handoff to a program coordinator — no images are ever stored, only the classification result

## Why This Matters

- **Zero connectivity required.** Every step — capture, inference, logging, the dashboard — works fully offline. No servers, no APIs, no internet permission requested anywhere in the app.
- **Cheap hardware.** A smartphone plus a ~$20–60 clip-on microscope lens attachment, not a lab.
- **Fits existing workflows.** Designed to slot into screening already being conducted by ASHA workers and PHC staff, not replace it.
- **Never overclaims.** Every result carries a persistent disclaimer, and positive/borderline results are explicitly routed toward lab confirmation — this is a triage aid, not a diagnostic authority.

## Model Performance

Both models use transfer learning on MobileNetV2, exported to TensorFlow Lite with float16 quantization (verified against int8 — float16 showed no meaningful accuracy loss for either disease).

| Metric | Sickle Cell | Malaria |
|---|---|---|
| Dataset size | 569 images (422 positive / 147 negative) | 27,558 images (balanced) |
| Test set size | 86 images | 4,134 images |
| Accuracy | 94.19% | 96.30% |
| Sensitivity (recall) | 93.75% | 94.34% |
| Specificity | 95.45% | 98.26% |
| Model size (TFLite, float16) | 4.6 MB | 4.6 MB |
| Referral threshold | <65% confidence → borderline/refer | <65% confidence → borderline/refer |

The 65% confidence threshold was independently validated for each model, not copied from one to the other — for malaria specifically, accuracy below the threshold is 58.4% (near coin-flip) versus 97.4% at or above it. Full breakdowns are in `model_output/*/results.md`.

**Datasets used:**
- Sickle cell: [Sickle Cell Disease Dataset](https://www.kaggle.com/datasets/florencetushabe/sickle-cell-disease-dataset) (Florence Tushabe, Kaggle)
- Malaria: [Malaria Cell Images Dataset](https://ceb.nlm.nih.gov/repositories/malaria-datasets/) (NIH/NLM, LHNCBC)

## Architecture

```
Capture (CameraX / Photo Picker)
        │
        ▼
Preprocess (resize, normalize — matched exactly to training pipeline)
        │
        ▼
   ┌────┴────┐
   ▼         ▼
Sickle    Malaria      ← modular, independently trained
 Cell      Model          per-disease TFLite models,
 Model                    sharing one capture/preprocess/
   │         │            refer pipeline
   └────┬────┘
        ▼
Confidence-based referral logic
        │
        ▼
Local logging (Room) → Dashboard + CSV export
```

Each disease is a self-contained model behind the same pipeline — adding a new disease means adding a new model and label set, not rebuilding the app.

## Tech Stack

- **Language:** Kotlin
- **Platform:** Native Android (min SDK 24+)
- **ML:** TensorFlow Lite (on-device inference), MobileNetV2 (transfer learning)
- **Camera:** CameraX
- **Gallery:** Android 13+ Photo Picker
- **Storage:** Room (local screening history — no images retained)
- **Model training:** Python, TensorFlow/Keras

## Project Structure

```
android/
├── app/src/main/java/com/sicklescan/app/
│   ├── MainActivity.kt              # UI, permissions, navigation
│   ├── CameraActivity.kt            # CameraX capture screen
│   ├── ImageClassifier.kt           # Model loading, preprocessing, inference
│   ├── Disease.kt                   # Per-disease model configuration
│   ├── ScreeningInterpreter.kt      # Confidence-based referral logic
│   ├── DashboardFragment.kt         # Screening history + stats UI
│   ├── DashboardAggregator.kt       # Per-disease stats aggregation
│   ├── CsvExporter.kt               # Screening log export
│   ├── data/                        # Room entities, DAO, database
│   └── BarChartView.kt              # 7-day trend chart (custom Canvas)
├── app/src/main/assets/             # Bundled .tflite models + labels
├── app/src/test/                    # Unit tests (pure-logic, no Android dependency)
└── app/src/androidTest/             # Instrumented tests (require a device/emulator)

model_output/
├── sickle_cell/                     # Trained model, results, confusion matrix
├── malaria/                         # Trained model, results, confusion matrix
└── comparison.md                    # Cross-model comparison table
```

## Getting Started

### Prerequisites
- Android Studio (recent stable version)
- An Android device or emulator running API 24+

### Build & Run
```bash
git clone https://github.com/Jeyxnth/SickleScan.git
cd android
./gradlew assembleDebug
```
Or open the `android/` folder directly in Android Studio and hit Run.

### Run Tests
```bash
./gradlew test                        # unit tests (no device needed)
./gradlew connectedDebugAndroidTest   # instrumented tests (device/emulator required)
```

### Download a Prebuilt APK
Prebuilt debug APKs are available under [Releases](../../releases). Since these aren't distributed via the Play Store, Android will prompt to allow "install from unknown sources" and may show a Play Protect warning — this is expected for any non-Play-Store APK, not a sign of a problem.

## Limitations & Honest Scope

This is a hackathon-stage prototype, not a clinically validated diagnostic tool:

- **Screening aid, not a diagnosis.** Every result routes positive/borderline cases to lab confirmation. This app does not and should not replace HPLC, solubility testing, or clinical judgment.
- **Trained on lab-quality images.** Both models were trained on clean, lab-microscope-captured images. Real-world accuracy on field images captured through a low-cost phone lens attachment — with variable lighting, focus, and staining — has not yet been measured and is expected to be lower.
- **No clinical validation.** Neither model has been reviewed by a hematologist or validated in a clinical setting.
- **Small sickle cell test set.** 86 images gives a real but fairly wide confidence interval — not a tight, production-grade validation.

## Future Scope

- **Thalassemia screening.** Deferred for now — no publicly available dataset with lab-confirmed (not just morphology-proxy) thalassemia labels was available at a usable size within project constraints. The [erythroSight dataset](https://www.frdr-dfdr.ca/repo/dataset/ca7ef0f8-28c2-4c3c-9f7c-3a30a8aac5f2) (Shrestha et al., 2024) is the identified path forward once storage/compute constraints allow.
- Field-image retraining once real phone-captured smear images are available
- Multi-device sync for coordinator-level coverage tracking (currently local-only, CSV export is the integration point)
- Proper Room migration path (current schema upgrade is destructive — acceptable pre-release, not for production)

## References

1. D. N. Breslauer, R. N. Maamari, N. A. Switz, W. A. Lam, and D. A. Fletcher, "Mobile phone based clinical microscopy for global health applications," *PLoS ONE*, vol. 4, no. 7, p. e6320, Jul. 2009.
2. A. Skandarajah, C. D. Reber, N. A. Switz, and D. A. Fletcher, "Quantitative imaging with a mobile phone microscope," *PLoS ONE*, vol. 9, no. 5, p. e96906, May 2014.
3. K. de Haan et al., "Automated screening of sickle cells using a smartphone-based microscope and deep learning," *npj Digital Medicine*, vol. 3, no. 1, art. 76, 2020.
4. Ministry of Health and Family Welfare, Government of India, "National Sickle Cell Anaemia Elimination Mission." [sickle.nhm.gov.in](https://sickle.nhm.gov.in/)
5. National Health Mission, "Guidelines on Hemoglobinopathies in India: Thalassemias, Sickle Cell Disease and Other Variant Hemoglobins," 2016.



---

*Built for MedVision'26. This is a prototype for educational and competition purposes — not certified or approved for clinical use.*
