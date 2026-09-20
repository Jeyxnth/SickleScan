# SickleScan Android app — Phase 2 + 3 + 4 + 6 + 7

Kotlin app that runs on-device models fully offline: capture or pick a
blood smear photo, choose which condition to screen for, tap Analyze, get
a screening result, confidence %, referral guidance, and the disclaimer —
never a diagnosis. Every screening is logged locally (Room, tagged with
which disease it was for) and can be reviewed as aggregate stats on a
Dashboard tab, or exported as CSV.

## Phase 7 additions (guardrail: reject non-smear images)

- **Pipeline**: capture -> preprocess -> **guardrail check** -> (smear-like) disease
  classifier -> result, or (not smear-like) -> warning. `guardrail_model.tflite`
  (float16, 4.7 MB) runs first via the same `ImageClassifier`; see
  `/model_output/guardrail/guardrail_results.md` for numbers and limits. It detects
  "matches the smear-like training distribution", NOT "blood smear vs other microscopy".
- **UX = soft warning** (a judgment call, chosen by the project owner): the message
  "This doesn't look like a blood smear photo. Please retake through the microscope
  attachment." with **Retake photo** as the main action and a smaller **Analyze
  anyway**. Reason: the guardrail's real-world false-reject rate is unknown, so a hard
  block could strand a legitimate-but-unusual photo. An overridden result shows a
  persistent caveat and is **not** logged as a disease screening.
- **Separate logging**: rejections go to their own Room table (`guardrail_events`,
  DB v3), never into disease stats. The Dashboard has an "Image check" section showing
  rejections / checks, rate, and how many were analyzed anyway.
- Threshold 0.5 is un-validated (the model's outputs are almost all 0 or 1) -- see results.
- `GuardrailClassifierInstrumentedTest` is build-verified only (no device available).

## Phase 6 additions (multi-disease: sickle cell + malaria)

Thalassemia is explicitly **out of scope** for this app: no lab-confirmed
public dataset was found at a usable size/quality (see the Phase 5 research
notes) -- it's not in the code, the UI, or the CSV schema.

- **Two bundled models**: `sicklescan_model.tflite` (unchanged) and the new
  `malaria_model.tflite` + `malaria_labels.txt`, trained in Phase 5 on the
  NIH/NLM LHNCBC Malaria Cell Images dataset (27,558 images) -- see
  `/model_output/malaria/malaria_results.md` and `/model_output/comparison.md`
  for full results (96.30% accuracy, 94.34% sensitivity, 98.26% specificity).
- **`Disease.kt`**: the set of conditions the app can screen for, each with
  its own model asset, labels asset, and Room storage key. `ImageClassifier`
  now takes a `Disease` and loads that model's actual input tensor shape at
  runtime -- not assumed to match sickle cell's just because both happen to
  be 224x224.
- **Disease selector**: a two-button toggle at the top of the capture screen
  ("Sickle Cell" / "Malaria"). No auto-detection of which disease an image
  is for -- that's a different, harder problem, out of scope here. Switching
  disease clears any result on screen, since a verdict for one disease
  doesn't apply to another.
- **Per-disease referral threshold, not copy-pasted**: malaria's 65%
  borderline ceiling was independently validated against its own test-set
  confidence distribution (58.4% accuracy below 65% confidence vs. 97.4%
  at/above it, only 2.7% of test images in the low-confidence band) --
  landing on the same number as sickle cell is a coincidence of the data,
  documented in `ScreeningInterpreter.kt` and `malaria_results.md`, not an
  assumption that one model's threshold transfers to another's.
- **Per-disease dashboard, not blended**: mixing sickle cell and malaria
  positives into one "% positive" figure would be actively misleading once
  the app screens for more than one condition, so `DashboardAggregator` now
  computes stats (`DiseaseStats`) separately per disease, and the Dashboard
  tab shows two sections. The CSV export gained a `disease` column.
- **Room schema bump** (v1 -> v2, `ScreeningRecord.disease`): uses
  `fallbackToDestructiveMigration()` since there are no shipped users yet --
  flagged in `AppDatabase.kt` as needing a real `Migration` before any real
  release.
- Still fully offline: no new permissions.

## Phase 4 additions

- **Local screening log (Room)**: after each analysis, `ScreenFragment`
  saves a record — timestamp, result (positive/borderline/negative),
  confidence %, referral flag. **No image is ever stored.** See
  `data/ScreeningRecord.kt` / `ScreeningDao.kt` / `AppDatabase.kt`.
- **Dashboard tab** (bottom nav, separate from the capture flow):
  total screenings, % positive/negative/borderline, referral count, and a
  positive-screenings-per-day bar chart for the last 7 days. All aggregation logic
  lives in `DashboardAggregator.kt` — pure Kotlin, no Room/Android
  dependency, so it's genuinely unit-tested here (4 passing tests).
- **Bar chart**: a small custom `View` (`BarChartView.kt`) drawn directly
  with `Canvas` — no charting library added, since a basic chart covered
  the need (per your instruction to ask before anything heavier; nothing
  heavier was needed).
- **CSV export**: `CsvExporter.kt` (pure Kotlin, 3 passing unit tests)
  builds the CSV; `DashboardFragment` writes it to a file and hands it to
  the Android share sheet (`ACTION_SEND` via `FileProvider`) — "Export CSV"
  → pick where it goes (save, email, etc.). This **is** the integration
  point with a coordinator's system in this scope; there's no server, and
  the app doesn't pretend otherwise.
- **Honesty note in the UI**: the Dashboard screen has a plain, always-
  visible line — "This data is local to this device only — it reflects
  this phone's screening activity, not a synced multi-worker view." Not a
  dialog, not a footnote; it's right under the title.
- Still fully offline: no new permissions, no network code anywhere.

## Phase 3 additions

- **No severity score.** Sliding-window/tile inference was considered and
  rejected: this model has only whole-image ground truth, no tile-level
  labels anywhere, so a "% tiles positive" number would be an unvalidated
  metric dressed up as a clinical one. Model confidence is shown instead,
  labeled plainly as confidence — see `ScreeningInterpreter.kt`'s file
  comment for the full reasoning. Doing real region-level scoring would
  need retraining with region-level labels, which per your instruction I
  flagged instead of building.
- **Referral logic** (`ScreeningInterpreter.kt`, pure Kotlin, no Android
  dependency): confidence < 65% → **Borderline**, "Uncertain result — refer
  for lab confirmation" regardless of which side of 50% the raw score
  landed on; confidence ≥ 65% and positive → **Positive**, "Refer for lab
  confirmation"; confidence ≥ 65% and negative → **Negative**, "Low risk —
  routine monitoring." Covered by 7 real, passing local unit tests
  (`./gradlew testDebugUnitTest` — these actually run in this environment,
  unlike the native-inference tests below).
- **UI polish**: Material 3 theme, a result card with distinct icon shapes
  (triangle/question-mark/checkmark) *and* colors per status (never color
  alone), and a loading state (shown for a minimum ~400ms so it's never a
  flash) while inference runs off the main thread.

## Build

```
cd android
./gradlew assembleDebug
```

Requires an installed Android SDK; `local.properties` here points at
`D:\AndroidSDK` (edit `sdk.dir` if yours lives elsewhere — this file is
gitignored since the path is machine-specific).

## Verifying predictions (11 held-out images: the original 8 + 3 new in Phase 3)

`app/src/androidTest/java/.../ImageClassifierInstrumentedTest.kt` loads the
bundled model exactly as the app does and checks its predictions against
known-correct labels for 11 held-out test images (bundled as test assets):
the original 8 from Phase 2 (all correctly predicted per
`/model_output/results.md`) plus 3 more added in Phase 3 — independently
verified in Python beforehand: `pos_21.jpg` (p=0.7256, a moderate-confidence
case — still a confident Positive under the 65% threshold, not Borderline),
`neg_140.jpg` (p=0.1248), `pos_393.jpg` (p=0.9999), all correct vs. ground
truth. It must run on a real
device or emulator — the TFLite native library only executes under the
real Android runtime/ABI, not a desktop JVM. Run it with a device/emulator
connected:

```
./gradlew connectedDebugAndroidTest
```

or in Android Studio: right-click the test class → Run.

**This environment had no Android emulator system image installed and no
physical device attached**, so I could not execute this instrumented test
myself — only build it. What I did verify here:
- `assembleDebug` and `assembleDebugAndroidTest` both build cleanly (no
  compile errors in `ImageClassifier`, `MainActivity`, `CameraActivity`, or
  `ScreeningInterpreter`).
- The exact preprocessing algorithm the Kotlin code uses (bilinear resize to
  the model's real input shape, `(pixel - 127.5) / 127.5` normalization) was
  cross-checked in Python against all 11 images and reproduced every
  actual/predicted label correctly.
- `ScreeningInterpreter`'s referral/borderline logic — the one piece of
  Phase 3 with no native dependency — has 7 real local unit tests, and they
  actually pass in this environment (`./gradlew testDebugUnitTest`).

Running the instrumented test above is the one remaining step to confirm
the real on-device native-inference path end to end — it should take a
couple of minutes on any connected device or emulator.

## Verifying the malaria model (Phase 6)

Same approach, same environment limitation: `MalariaClassifierInstrumentedTest.kt`
runs `ImageClassifier(context, Disease.MALARIA)` against 8 test images
under `app/src/androidTest/assets/malaria_test_images/`, checked against
predictions independently verified in Python against the actual
`.tflite` file. One of the 8 (`parasitized_misclassified.png`) is a real
model error (predicts uninfected for a truly parasitized cell) kept
deliberately rather than cherry-picked away — the test asserts against
what the model actually predicts, so it verifies the Kotlin
implementation faithfully reproduces the model's real behavior, errors
included, not that the model is always right. Same limitation as
above: build-verified here, not run on a device.

## Verifying the log/dashboard/CSV (Phase 4)

Room's SQLite backing also only runs under the real Android runtime, not a
desktop JVM, so — same limitation as the TFLite native library — I could
not run the app itself here to click through 11 screenings by hand. What I
did instead: `EndToEndLogSimulationTest.kt` feeds the same 11 real,
independently-verified probabilities above through the exact
`ScreeningInterpreter → ScreeningRecord → DashboardAggregator → CsvExporter`
pipeline the app uses, and prints the resulting CSV. Run it yourself with:

```
./gradlew testDebugUnitTest --tests "*EndToEndLogSimulationTest*" -i
```

Result from the last run (9 positive, 2 negative, 0 borderline — none of
these 11 real images happen to land in the 50-65% borderline band). CSV
now includes the `disease` column added in Phase 6:

```
timestamp,disease,result,confidence_percent,referral_flag
2026-09-17T18:54:01,sickle_cell,positive,99.1,yes
2026-09-17T18:54:01,sickle_cell,negative,98.2,no
2026-09-18T18:54:01,sickle_cell,positive,96.5,yes
2026-09-18T18:54:01,sickle_cell,positive,99.8,yes
2026-09-18T18:54:01,sickle_cell,positive,100.0,yes
2026-09-19T18:54:01,sickle_cell,positive,99.3,yes
2026-09-19T18:54:01,sickle_cell,positive,99.1,yes
2026-09-19T18:54:01,sickle_cell,negative,87.5,no
2026-09-20T18:54:01,sickle_cell,positive,98.5,yes
2026-09-20T18:54:01,sickle_cell,positive,97.9,yes
2026-09-20T18:54:01,sickle_cell,positive,72.6,yes

total=11 positive=9 negative=2 borderline=0 referrals=9
positive%=81.8 negative%=18.2
```

To confirm the actual UI (Dashboard numbers, bar chart, real Room writes,
the share-sheet CSV file) on a device: install the app, run Analyze on a
handful of the sample images (available under
`app/src/androidTest/assets/sample_test_images/` — push them to the
device/emulator gallery first), switch to the Dashboard tab, and tap
Export CSV.

## Notes

- TFLite input tensor is read from each model at load time (not
  hardcoded/assumed): both the sickle cell and malaria models happen to be
  float32, `[1, 224, 224, 3]`, no built-in quantization, since both were
  trained with the identical Phase 1/5 pipeline — but `ImageClassifier`
  doesn't assume that; it asks the interpreter.
- No `INTERNET` permission is declared; nothing in the app makes network
  calls.
- Gallery picking uses the system Photo Picker
  (`ActivityResultContracts.PickVisualMedia`), which needs no storage
  permission on any supported API level; the manifest's
  `READ_EXTERNAL_STORAGE` (capped at API 32) is a defensive fallback only.
- Camera capture is a small dedicated `CameraActivity` (CameraX
  `Preview` + `ImageCapture`) that returns the photo's `content://` Uri via
  `FileProvider`.
- The Room database (`sicklescan.db`) lives in the app's private storage;
  CSV exports are written to the app's private cache dir and shared out
  only via `FileProvider` + the share sheet — no new permissions needed for
  either.
