# MobileNetV2 image inference

The host preprocessor applies five variants to every supported image in `imgs/`:
original, grayscale, reduced brightness, increased contrast, and sharpened.
It saves viewable PNGs under `filtered_images/` and matching 224x224 model input
tensors beside them. MobileNetV2 is a classifier, so its outputs are class
scores rather than generated images.

## Prepare inputs and filtered image outputs

From the repository root, install Pillow and generate the inputs:

```powershell
python -m pip install Pillow
python tools\prepare_images.py --output build\thermal_inputs
```

The filtered images are in `build\thermal_inputs\filtered_images`; `manifest.json`
maps each output back to its source and filter. The `.tensor` files are raw
little-endian float32 NCHW values. Preprocessing scales RGB values to `[-1, 1]`;
confirm that this matches the specific ONNX export before interpreting scores.

## Run the thermal-aware daemon

Build the Android target with the NDK as usual, then copy the daemon, model, and
input directory to the device:

```powershell
adb push build-a2core\thermal_daemon /data/local/tmp/thermal_daemon
adb push third_party\onnx\lib\armeabi-v7a\libonnxruntime.so /data/local/tmp/libonnxruntime.so
adb push third_party\onnx\lib\libc++_shared.so /data/local/tmp/libc++_shared.so
adb push third_party\onnx\mobilenetv2.onnx /data/local/tmp/mobilenetv2.onnx
adb push third_party\onnx\mobilenetv2.onnx.data /data/local/tmp/mobilenetv2.onnx.data
adb push build\thermal_inputs /data/local/tmp/thermal_inputs
adb shell chmod +x /data/local/tmp/thermal_daemon
adb shell "cd /data/local/tmp; LD_LIBRARY_PATH=/data/local/tmp ./thermal_daemon"
```

The shared libraries must match the executable's Android ABI. The commands
above use `armeabi-v7a` for the Samsung A2 Core. `LD_LIBRARY_PATH` tells
Android's dynamic linker to look in `/data/local/tmp` for ONNX Runtime and the
shared C++ runtime.

The daemon runs for five minutes, cycles through the tensors, runs classification,
and switches between NNAPI and CPU according to its thermal thresholds. It
prints each prediction and the completed-cycle count, and updates
`/data/local/tmp/thermal_results.csv` after each complete cycle. It stops at the
five-minute deadline (after any inference already in progress finishes) or when
you press Ctrl+C. Pull the latest report with:

```powershell
adb pull /data/local/tmp/thermal_results.csv .
```

## Explicit backend-switch benchmark

The five-minute limit applies to normal daemon mode. The benchmark instead runs
three full passes (NNAPI, CPU, then NNAPI) and exits.

To run all tensors on NNAPI, CPU, and NNAPI again:

```powershell
adb shell "cd /data/local/tmp; LD_LIBRARY_PATH=/data/local/tmp ./thermal_daemon --benchmark /data/local/tmp/thermal_inputs"
adb pull /data/local/tmp/thermal_benchmark_results.csv .
```

The benchmark fails if it cannot load NNAPI, a requested backend switch does not
take effect, a tensor is malformed, or the execution profile contains no events
for the expected provider. Profile JSON files are written under
`/data/local/tmp/thermal_daemon_profile_*`.

NNAPI profiling verifies NNAPI node execution, not a specific GPU. Depending on
the Android version, device, model partitioning, and vendor driver, execution
may use a GPU, NPU, another accelerator, or CPU fallback. Use vendor profiling
tools if proving physical GPU utilization is required.
