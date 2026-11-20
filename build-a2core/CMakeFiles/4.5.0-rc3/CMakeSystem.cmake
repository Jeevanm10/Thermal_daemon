set(CMAKE_HOST_SYSTEM "Windows-10.0.26300")
set(CMAKE_HOST_SYSTEM_NAME "Windows")
set(CMAKE_HOST_SYSTEM_VERSION "10.0.26300")
set(CMAKE_HOST_SYSTEM_PROCESSOR "AMD64")

set(_CMAKE_INPUT_TOOLCHAIN_FILE [==[C:\android-ndk-r30\build\cmake\android.toolchain.cmake]==])
set(_CMAKE_SYSTEM_TOOLCHAIN_FILE "C:/android-ndk-r30/build/cmake/android.toolchain.cmake")
include("${_CMAKE_SYSTEM_TOOLCHAIN_FILE}")

set(CMAKE_SYSTEM "Android-1")
set(CMAKE_SYSTEM_NAME "Android")
set(CMAKE_SYSTEM_VERSION "1")
set(CMAKE_SYSTEM_PROCESSOR "armv7-a")

set(CMAKE_CROSSCOMPILING "TRUE")

set(CMAKE_SYSTEM_LOADED 1)
