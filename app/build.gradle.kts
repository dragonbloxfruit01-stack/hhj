plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "ai.localai800"
    compileSdk = 35

    defaultConfig {
        applicationId = "ai.localai800"
        minSdk = 28
        targetSdk = 35
        versionCode = 3
        versionName = "0.3.0-training-foundation"
        ndkVersion = "27.0.12077973"
        ndk { abiFilters += listOf("arm64-v8a") }
        externalNativeBuild {
            cmake { cppFlags += listOf("-std=c++17", "-O3") }
        }
    }

    externalNativeBuild {
        cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    implementation("com.mobilefinetuner:mobilefinetuner-android:0.1.0")
}
