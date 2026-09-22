package com.sicklescan.app.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

// v5 added ScreeningRecord.wideField + cellsDetected (Phase 13, wide-field malaria) via a real Migration, so existing
// logs survive the update. Only versions older than 4 still fall back to a destructive rebuild.
// v4 added capture_sessions + the ScreeningRecord.sessionId foreign key (Phase 8).
// v3 added guardrail_events (Phase 7). v2 added ScreeningRecord.disease (Phase 6).
// Versions before 4 predate any shipped build, so they still use a destructive fallback.
@Database(
    entities = [CaptureSession::class, ScreeningRecord::class, GuardrailEvent::class],
    version = 5,
    exportSchema = false,
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun screeningDao(): ScreeningDao
    abstract fun guardrailDao(): GuardrailDao

    companion object {
        /** Phase 13: numeric defaults only (0), matching the entity's @ColumnInfo(defaultValue = "0"). */
        val MIGRATION_4_5 = object : Migration(4, 5) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE screening_records ADD COLUMN wideField INTEGER NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE screening_records ADD COLUMN cellsDetected INTEGER NOT NULL DEFAULT 0")
            }
        }

        @Volatile
        private var instance: AppDatabase? = null

        fun getInstance(context: Context): AppDatabase =
            instance ?: synchronized(this) {
                instance ?: Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "sicklescan.db"
                )
                    .addMigrations(MIGRATION_4_5)
                    .fallbackToDestructiveMigration()
                    .build().also { instance = it }
            }
    }
}
