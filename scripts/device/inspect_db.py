"""Read-only inspection of a pulled sicklescan.db (Room). Usage: python scripts/device/inspect_db.py PATH_TO_DB_DIR_OR_FILE
Copies the db (+ -wal/-shm) to a temp folder first so the original backup is never modified."""
import os
import shutil
import sqlite3
import sys
import tempfile

src = sys.argv[1]
d = src if os.path.isdir(src) else os.path.dirname(src)
tmp = tempfile.mkdtemp()
for f in ("sicklescan.db", "sicklescan.db-wal", "sicklescan.db-shm"):
    if os.path.exists(os.path.join(d, f)):
        shutil.copy(os.path.join(d, f), os.path.join(tmp, f))
con = sqlite3.connect(os.path.join(tmp, "sicklescan.db"))
print("user_version:", con.execute("PRAGMA user_version").fetchone()[0])
for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'android_%' ORDER BY name"):
    cols = [(r[1], r[2], r[3], r[4]) for r in con.execute(f"PRAGMA table_info({t})")]
    n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"{t}: {n} rows; columns (name, type, notnull, default): {cols}")
for name in ("screening_records", "capture_sessions", "guardrail_events"):
    try:
        rows = con.execute(f"SELECT * FROM {name} ORDER BY id").fetchall()
    except sqlite3.Error:
        continue
    import hashlib
    print(f"{name} content hash (first 8 columns of every row): {hashlib.sha256(repr([r[:8] for r in rows]).encode()).hexdigest()[:16]}")
if con.execute("SELECT name FROM sqlite_master WHERE name='screening_records'").fetchone():
    print("results by disease/result:", con.execute("SELECT disease, result, COUNT(*) FROM screening_records GROUP BY 1,2").fetchall())
    print("sessions by guardrailResult:", con.execute("SELECT guardrailResult, COUNT(*) FROM capture_sessions GROUP BY 1").fetchall())
