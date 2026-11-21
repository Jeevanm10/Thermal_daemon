# MobileNetV2 image inference

The host preprocessor applies five variants to every supported image in `imgs/`:
original, grayscale, reduced brightness, increased contrast, and sharpened.
It saves viewable PNGs under `filtered_images/` and matching 224x224 model input
tensors beside them. MobileNetV2 is a classifier, so its outputs are class
scores rather than generated images.

## Prepare inputs and the TensorFlow Lite model

Generate the image inputs from the repository root:

```powershell
python -m pip install Pillow
python tools\prepare_images.py --output build\thermal_inputs
```

The `.tensor` files are raw little-endian float32 NCHW values, with RGB values
scaled to `[-1, 1]`. The native daemon now uses TensorFlow Lite (LiteRT), not
ONNX Runtime, and loads `/data/local/tmp/mobilenetv2.tflite`. The existing
`third_party/onnx/mobilenetv2.onnx` cannot be loaded directly by TensorFlow Lite:
convert it to a float32 `.tflite` model separately (for example, with
[onnx2tf](https://github.com/PINTO0309/onnx2tf)). The converter may need
additional model-specific fixes; verify conversion output against the ONNX
model before using it. Copy the validated float32 model to
`build/mobilenetv2.tflite`. The daemon accepts model
inputs shaped `[1,224,224,3]` (NHWC) or `[1,3,224,224]` (NCHW), and transposes
the prepared tensors when the model uses NHWC. Check that the conversion
preserves the model's `[-1, 1]` input preprocessing and produces float32 output.

An example conversion invocation is:

```powershell
python -m pip install onnx2tf
onnx2tf -i third_party\onnx\mobilenetv2.onnx -o build\mobilenetv2_converted
```

Inspect the converter output and select the validated float32 model; do not
assume every exported variant has the right input layout or preprocessing.

For the Samsung A2 Core build, provide a LiteRT/TensorFlow Lite C++ distribution
in this layout:

```text
third_party/tflite/
  include/tensorflow/lite/interpreter.h
  include/tensorflow/lite/delegates/gpu/delegate.h
  lib/armeabi-v7a/libtensorflowlite.so
  lib/armeabi-v7a/libtensorflowlite_gpu_delegate.so
  lib/armeabi-v7a/libc++_shared.so
```

Configure and build with the Android NDK:

```powershell
cmake -S . -B build-a2core -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE="$env:ANDROID_NDK_HOME\build\cmake\android.toolchain.cmake" `
  -DANDROID_ABI=armeabi-v7a `
  -DANDROID_PLATFORM=android-24 `
  -DBUILD_FOR_SAMSUNG_A2_CORE=ON `
  -DTFLITE_ROOT="$PWD\third_party\tflite"
cmake --build build-a2core
```

## Run the thermal-aware daemon

Copy the executable, converted model, LiteRT libraries, C++ runtime, and input
tensors to the phone. The shared libraries must match the executable's
`armeabi-v7a` ABI.

```powershell
adb push build-a2core\thermal_daemon /data/local/tmp/thermal_daemon
adb push build\mobilenetv2.tflite /data/local/tmp/mobilenetv2.tflite
adb push third_party\tflite\lib\armeabi-v7a\libtensorflowlite.so /data/local/tmp/
adb push third_party\tflite\lib\armeabi-v7a\libtensorflowlite_gpu_delegate.so /data/local/tmp/
adb push third_party\tflite\lib\armeabi-v7a\libc++_shared.so /data/local/tmp/
adb push build\thermal_inputs /data/local/tmp/thermal_inputs
adb shell chmod +x /data/local/tmp/thermal_daemon
adb shell "cd /data/local/tmp; LD_LIBRARY_PATH=/data/local/tmp ./thermal_daemon"
```

Normal startup requires creating and applying the TensorFlow Lite GPU delegate;
it exits with an error instead of silently starting on CPU if that fails. The
thermal policy switches to the CPU backend at configured thresholds and
restores the GPU delegate after recovery. Results are written to
`/data/local/tmp/thermal_results.csv`.

The GPU delegate may assign unsupported operators to CPU, so attaching the
delegate does not guarantee every model operation ran on the GPU. Availability
also depends on the phone's Android version, OpenGL ES support, GPU driver, and
the converted model's operators. Check Android logs for delegate warnings and
use vendor profiling tools to verify physical GPU execution.

## Backend-switch benchmark

The benchmark runs three full passes (GPU delegate, CPU, then GPU delegate):

```powershell
adb shell "cd /data/local/tmp; LD_LIBRARY_PATH=/data/local/tmp ./thermal_daemon --benchmark /data/local/tmp/thermal_inputs"
adb pull /data/local/tmp/thermal_benchmark_results.csv .
```

It verifies that the requested backend transitions succeeded and that
inference produced outputs. It does not prove that the GPU executed every
operator; use the device's profiling tools for that.
