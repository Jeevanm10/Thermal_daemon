cmake_minimum_required(VERSION 3.15)
project(ThermalDaemon CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_FOR_SAMSUNG_A2_CORE
    "Build for the Samsung Galaxy A2 Core (32-bit ARM/armeabi-v7a)"
    OFF)

if(ANDROID)
    set(ANDROID_STL "c++_shared" CACHE STRING "Android C++ runtime")
    if(NOT ANDROID_PLATFORM)
        set(ANDROID_PLATFORM "android-24" CACHE STRING "Minimum Android API")
    endif()
endif()

set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnx")
set(ANDROID_ABI_VALUE "")

if(ANDROID)
    if(BUILD_FOR_SAMSUNG_A2_CORE)
        set(ANDROID_ABI_VALUE "armeabi-v7a")
        if(ANDROID_ABI AND NOT ANDROID_ABI STREQUAL "armeabi-v7a")
            message(FATAL_ERROR
                "Samsung A2 Core build requires ANDROID_ABI=armeabi-v7a "
                "(requested '${ANDROID_ABI}')")
        endif()
    elseif(ANDROID_ABI)
        set(ANDROID_ABI_VALUE "${ANDROID_ABI}")
    else()
        set(ANDROID_ABI_VALUE "armeabi-v7a")
    endif()
else()
    set(ANDROID_ABI_VALUE "arm64-v8a")
endif()

set(ONNXRUNTIME_LIBRARY
    "${ONNXRUNTIME_ROOT}/lib/${ANDROID_ABI_VALUE}/libonnxruntime.so")

if(ANDROID AND EXISTS "${ONNXRUNTIME_LIBRARY}")
    add_executable(thermal_daemon
        src/main.cpp
        src/ThermalMonitor.cpp
        src/InferenceScheduler.cpp
    )

    target_link_libraries(thermal_daemon PRIVATE log dl)
    target_include_directories(thermal_daemon PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${ONNXRUNTIME_ROOT}/include"
    )

    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    )

    target_link_libraries(thermal_daemon PRIVATE onnxruntime dl)
elseif(NOT ANDROID)
    message(STATUS
        "Skipping thermal_daemon: the inference target requires an Android NDK build")
else()
    message(FATAL_ERROR
        "Missing ONNX Runtime library for Android ABI '${ANDROID_ABI_VALUE}': "
        "${ONNXRUNTIME_LIBRARY}")
endif()