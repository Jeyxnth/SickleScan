# SickleScan Android app — Phase 2 + 3 + 4

Kotlin app that runs the Phase 1 `sicklescan_model.tflite` fully on-device: capture or
pick a blood smear photo, tap Analyze, get a screening result, confidence %,
referral guidance, and the disclaimer — never a diagnosis. Every screening
is logged locally (Room) and can be reviewed as aggregate stats on a
Dashboard tab, or exported as CSV.

## Phase 4 additions

- **Local screening log (Room)**: after each analysis, `ScreenFragment`
  saves a record — timestamp, result (positive/borderline/negative),
  confidence %, referral flag. **No image is ever stored.** See
  `data/ScreeningRecord.kt` / `ScreeningDao.kt` / `AppDatabase.kt`.
- **Dashboard tab** (bottom nav, separate from the capture flow):
  total screenings, % positive/negative/borderline, referral count, and a
  screenings-per-day bar chart for the last 7 days. All aggregation logic
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
these 11 real images happen to land in the 50-65% borderline band):

```
timestamp,result,confidence_percent,referral_flag
2026-09-14T11:01:48,positive,99.1,yes
2026-09-14T11:01:48,negative,98.2,no
2026-09-15T11:01:48,positive,96.5,yes
2026-09-15T11:01:48,positive,99.8,yes
2026-09-15T11:01:48,positive,100.0,yes
2026-09-16T11:01:48,positive,99.3,yes
2026-09-16T11:01:48,positive,99.1,yes
2026-09-16T11:01:48,negative,87.5,no
2026-09-17T11:01:48,positive,98.5,yes
2026-09-17T11:01:48,positive,97.9,yes
2026-09-17T11:01:48,positive,72.6,yes

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

- TFLite input tensor was inspected directly (not assumed): float32,
  shape `[1, 224, 224, 3]`, no built-in quantization — matches
  `ImageClassifier.kt`'s preprocessing.
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
