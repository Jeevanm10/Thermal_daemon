cmake_minimum_required(VERSION 3.15)
project(ThermalDaemon CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(thermal_monitor STATIC
    src/ThermalMonitor.cpp
)
target_include_directories(thermal_monitor PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# 1. Target WITHOUT Inference (Pure Monitoring & Logging)
add_executable(thermal_monitor_only
    src/main.cpp
)
target_link_libraries(thermal_monitor_only PRIVATE thermal_monitor)

# 2. Target WITH Inference (Full MobileNet Scheduler)
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnx")
set(ANDROID_ABI_VALUE "${ANDROID_ABI}")
if(NOT ANDROID_ABI_VALUE)
    set(ANDROID_ABI_VALUE "arm64-v8a")
endif()

set(ONNXRUNTIME_LIBRARY
    "${ONNXRUNTIME_ROOT}/lib/${ANDROID_ABI_VALUE}/libonnxruntime.so")

if(ANDROID AND EXISTS "${ONNXRUNTIME_LIBRARY}")
    add_executable(thermal_daemon
        src/main.cpp
        src/InferenceScheduler.cpp
    )

    target_compile_definitions(thermal_daemon PRIVATE THERMAL_INFERENCE_ENABLED)
    target_link_libraries(thermal_daemon PRIVATE thermal_monitor)
    target_include_directories(thermal_daemon PRIVATE
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