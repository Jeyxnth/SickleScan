package com.sicklescan.app.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

// v3 added the guardrail_events table (Phase 7); same destructive-fallback caveat as v2.
// v2 added ScreeningRecord.disease (multi-disease support, Phase 6). The
// app has no shipped users/backward-compat requirement yet, so this uses
// a destructive fallback rather than a real Migration -- acceptable
// pre-release; revisit with a proper Migration before any real release.
@Database(entities = [ScreeningRecord::class, GuardrailEvent::class], version = 3, exportSchema = false)
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
