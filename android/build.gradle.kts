// Top-level build file
plugins {
    id("com.android.application") version "8.5.2" apply false
    id("org.jetbrains.kotlin.android") version "1.9.24" apply false
    // KSP for Room's annotation processor -- lighter/faster than kapt.
    id("com.google.devtools.ksp") version "1.9.24-1.0.20" apply false
}
