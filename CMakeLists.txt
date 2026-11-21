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

set(TFLITE_ROOT "${CMAKE_SOURCE_DIR}/third_party/tflite" CACHE PATH
    "LiteRT/TensorFlow Lite C++ distribution with Android libraries")
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

set(TFLITE_LIBRARY
    "${TFLITE_ROOT}/lib/${ANDROID_ABI_VALUE}/libtensorflowlite.so")
set(TFLITE_GPU_DELEGATE_LIBRARY
    "${TFLITE_ROOT}/lib/${ANDROID_ABI_VALUE}/libtensorflowlite_gpu_delegate.so")

if(ANDROID AND
   EXISTS "${TFLITE_ROOT}/include/tensorflow/lite/interpreter.h" AND
   EXISTS "${TFLITE_ROOT}/include/tensorflow/lite/model.h" AND
   EXISTS "${TFLITE_ROOT}/include/tensorflow/lite/kernels/register.h" AND
   EXISTS "${TFLITE_ROOT}/include/tensorflow/lite/delegates/gpu/delegate.h" AND
   EXISTS "${TFLITE_LIBRARY}" AND
   EXISTS "${TFLITE_GPU_DELEGATE_LIBRARY}")
    add_executable(thermal_daemon
        src/main.cpp
        src/ThermalMonitor.cpp
        src/InferenceScheduler.cpp
    )

    target_link_libraries(thermal_daemon PRIVATE
        log dl android EGL GLESv2
    )
    target_include_directories(thermal_daemon PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${TFLITE_ROOT}/include"
    )

    add_library(tensorflowlite SHARED IMPORTED)
    set_target_properties(tensorflowlite PROPERTIES
        IMPORTED_LOCATION "${TFLITE_LIBRARY}"
    )
    add_library(tensorflowlite_gpu_delegate SHARED IMPORTED)
    set_target_properties(tensorflowlite_gpu_delegate PROPERTIES
        IMPORTED_LOCATION "${TFLITE_GPU_DELEGATE_LIBRARY}"
    )

    target_link_libraries(thermal_daemon PRIVATE
        tensorflowlite_gpu_delegate
        tensorflowlite
    )
elseif(NOT ANDROID)
    message(STATUS
        "Skipping thermal_daemon: the inference target requires an Android NDK build")
else()
    message(FATAL_ERROR
        "Missing LiteRT/TensorFlow Lite C++ headers or libraries for Android "
        "ABI '${ANDROID_ABI_VALUE}'. Expected headers under "
        "'${TFLITE_ROOT}/include' and libraries "
        "'${TFLITE_LIBRARY}' and '${TFLITE_GPU_DELEGATE_LIBRARY}'.")
endif()