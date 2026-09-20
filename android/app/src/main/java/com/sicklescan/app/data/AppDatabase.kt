package com.sicklescan.app.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

// v4 added capture_sessions + the ScreeningRecord.sessionId foreign key (Phase 8).
// v3 added guardrail_events (Phase 7). v2 added ScreeningRecord.disease (Phase 6).
// The app has no shipped users/backward-compat requirement yet, so this uses a
// destructive fallback rather than a real Migration -- acceptable pre-release;
// revisit with a proper Migration before any real release.
@Database(
    entities = [CaptureSession::class, ScreeningRecord::class, GuardrailEvent::class],
    version = 4,
    exportSchema = false,
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun screeningDao(): ScreeningDao
    abstract fun guardrailDao(): GuardrailDao

    companion object {
        @Volatile
        private var instance: AppDatabase? = null

        fun getInstance(context: Context): AppDatabase =
            instance ?: synchronized(this) {
                instance ?: Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "sicklescan.db"
                )
                    .fallbackToDestructiveMigration()
                    .build().also { instance = it }
            }
    }
}
