"""Row-by-row comparison of the pulled DB before and after the Room migration 4->5 (works on temp copies; originals untouched).
Usage: python scripts/device/compare_dbs.py BEFORE_DIR AFTER_DIR"""
import os
import shutil
import sqlite3
import sys
import tempfile


def open_copy(d):
    tmp = tempfile.mkdtemp()
    for f in ("sicklescan.db", "sicklescan.db-wal", "sicklescan.db-shm"):
        if os.path.exists(os.path.join(d, f)):
            shutil.copy(os.path.join(d, f), os.path.join(tmp, f))
    return sqlite3.connect(os.path.join(tmp, "sicklescan.db"))


b, a = open_copy(sys.argv[1]), open_copy(sys.argv[2])
ok = True
for table in ("capture_sessions", "guardrail_events", "screening_records"):
    cols_b = [r[1] for r in b.execute(f"PRAGMA table_info({table})")]
    rb = b.execute(f"SELECT {','.join(cols_b)} FROM {table} ORDER BY id").fetchall()
    ra = a.execute(f"SELECT {','.join(cols_b)} FROM {table} ORDER BY id").fetchall()   # only the ORIGINAL columns, after migration
    same = rb == ra
    ok &= same
    print(f"{table}: before {len(rb)} rows, after {len(ra)} rows, original columns identical row-by-row: {same}")
new = a.execute("SELECT wideField, cellsDetected, COUNT(*) FROM screening_records GROUP BY 1,2").fetchall()
print("new columns (wideField, cellsDetected, rows):", new)
ok &= new == [(0, 0, len(rb))]
print("foreign key violations after migration:", a.execute("PRAGMA foreign_key_check").fetchall())
print("integrity_check:", a.execute("PRAGMA integrity_check").fetchall())
print("RESULT:", "PASS" if ok else "FAIL")
