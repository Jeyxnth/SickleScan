package com.sicklescan.app.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query

@Dao
interface ScreeningDao {

    @Insert
    suspend fun insert(record: ScreeningRecord)

    // Records are fetched in full and aggregated in Kotlin (DashboardStats),
    // not via SQL aggregate queries -- simpler and just as fast at the
    // scale a single device's screening log runs at, and it keeps the
    // aggregation logic in one place that's easy to unit test.
    @Query("SELECT * FROM screening_records ORDER BY timestampMillis DESC")
    suspend fun getAll(): List<ScreeningRecord>
}
