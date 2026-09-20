package com.sicklescan.app.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import androidx.room.Transaction

@Dao
abstract class ScreeningDao {

    @Insert
    abstract suspend fun insertSession(session: CaptureSession): Long

    @Insert
    abstract suspend fun insertRecord(record: ScreeningRecord): Long

    /** Writes a session and all its records atomically, so a session is never
     * saved with only some of its records. Returns the new session id. */
    @Transaction
    open suspend fun insertSessionWithRecords(session: CaptureSession, records: List<ScreeningRecord>): Long {
        val sessionId = insertSession(session)
        records.forEach { insertRecord(it.copy(sessionId = sessionId)) }
        return sessionId
    }

    // Records/sessions are fetched in full and aggregated in Kotlin (DashboardStats),
    // not via SQL aggregate queries -- simpler and just as fast at the scale a single
    // device's log runs at, and it keeps the aggregation logic in one unit-tested place.
    @Query("SELECT * FROM screening_records ORDER BY timestampMillis DESC")
    abstract suspend fun getAllRecords(): List<ScreeningRecord>

    @Query("SELECT * FROM capture_sessions ORDER BY timestampMillis DESC")
    abstract suspend fun getAllSessions(): List<CaptureSession>
}
