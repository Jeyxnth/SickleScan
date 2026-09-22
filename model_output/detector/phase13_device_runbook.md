# Phase 13 — on-device session runbook (all three checks, one session)

**Status: RUN on a Motorola Edge 50 Pro; results are in `phase13_report.md`, section "On-device session". Notes for a re-run: use `am instrument` directly (Gradle's connected-test task uninstalls the app and would wipe its data), and set `MSYS_NO_PATHCONV=1` in Git Bash when pushing to /sdcard.** Everything below is prepared so the session takes minutes once a phone is connected (USB debugging on, "Allow USB debugging" accepted). `ADB=/d/AndroidSDK/platform-tools/adb.exe`.

Staged locally (gitignored): `demo_images/device_checks/app-v4-baseline.apk` (last committed build, Room DB v4) and `app-phase13.apk` (current build, DB v5); `demo_images/wide_field_eval/` (12 infected + 8 negative BBBC041 **validation** fields + `expected.csv`). Both APKs are debug builds (same debug keystore), so an in-place upgrade works on a phone that has only debug-signed builds.

## Before anything: protect real data
The phone may hold real screening logs. **Never `adb uninstall` and do not "Clear storage" without asking.** If installing over the existing app fails with `INSTALL_FAILED_UPDATE_INCOMPATIBLE` (different signing key), stop and ask; use a second device/emulator or export the CSV first.

## a. Latency and phone-vs-Python comparison
```
$ADB install -r demo_images/device_checks/app-phase13.apk
$ADB push demo_images/wide_field_eval /sdcard/Android/data/com.sicklescan.app/files/wide_field_eval
$ADB logcat -c
cd android && ./gradlew connectedDebugAndroidTest --tests "*WideFieldPipelineInstrumentedTest"
$ADB logcat -d -s WideFieldEval
```
Report: warm-up run, per-field total, median/mean/max, mean per stage (prepare / detect / crop / classify), plus "decision mismatches vs Python" (compares the FULL answer, Positive / Negative / Inconclusive, including the 20-cell count gate), top-score differences > 0.05, cell counts differing by > 2. Pass: 0 answer mismatches, at most 10% top-score differences. Also time one field photo by hand in the app (the result card shows "took X s").

## b. Latency rule
If the median per-field total is several seconds or more, implement crop batching for the classifier calls (resize the classifier input to [N,224,224,3] and run crops in groups; no score is changed, so no re-validation beyond re-running (a)). **Do not switch to int8 without approval** (it changes validated scores and needs the float16-vs-int8 re-validation).

## c. Room migration 4 -> 5 with existing data
1. Install the baseline and run it once so it creates a v4 database: `$ADB install -r demo_images/device_checks/app-v4-baseline.apk`; open the app, then `$ADB shell am force-stop com.sicklescan.app`.
2. Pull the database (debuggable build): `$ADB exec-out run-as com.sicklescan.app cat databases/sicklescan.db > v4.db` (also `sicklescan.db-wal` / `-shm` if present).
3. Seed known rows on the PC with Python `sqlite3` (a few `capture_sessions` and `screening_records` rows, both diseases, one `overridden` session; note the counts and values), confirm `PRAGMA user_version` is 4, push it back (`$ADB push v4.db /data/local/tmp/` then `run-as ... cp /data/local/tmp/v4.db databases/sicklescan.db`, removing stale -wal/-shm).
4. Upgrade in place: `$ADB install -r demo_images/device_checks/app-phase13.apk`, launch the app (this opens the DB and runs Migration 4->5), check `adb logcat` for any `IllegalStateException` / "Migration didn't properly handle".
5. Pull the DB again: expect `PRAGMA user_version = 5`, `screening_records` has `wideField` and `cellsDetected` with value 0 for every old row, row counts and values identical to what was seeded, foreign keys intact. In the app, the Dashboard totals and an exported CSV must match the seeded data (old rows show `image,0` in the new CSV columns).
6. Then run one wide-field analysis on the phone and confirm the new row has `wideField = 1`, the cell count, and (for a wide-field-only run) session image check `skipped`.
Pass: app opens, no crash, all seeded rows intact, new columns default to 0. Any migration exception is a ship blocker.

## Do not
Run the four reserved user photos; commit anything before all three checks are reported; use int8.
