#include "InferenceScheduler.hpp"

#include <tensorflow/lite/delegates/gpu/delegate.h>
#include <tensorflow/lite/kernels/register.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace {
constexpr size_t kInputSize = 3 * 224 * 224;
}

InferenceScheduler::InferenceScheduler(const std::string& modelPath)
    : model_(tflite::FlatBufferModel::BuildFromFile(modelPath.c_str())) {
    if (!model_) {
        throw std::runtime_error("could not load TensorFlow Lite model: " +
                                 modelPath);
    }
    if (!initializeSession(ExecutionBackend::GPU_DELEGATE)) {
        throw std::runtime_error(
            "TensorFlow Lite GPU delegate could not initialize; refusing to "
            "start on CPU");
    }
}

InferenceScheduler::~InferenceScheduler() {
    interpreter_.reset();
    if (gpuDelegate_) {
        TfLiteGpuDelegateV2Delete(gpuDelegate_);
    }
}

bool InferenceScheduler::initializeSession(ExecutionBackend backend) {
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> nextInterpreter;
    const TfLiteStatus buildStatus =
        tflite::InterpreterBuilder(*model_, resolver)(&nextInterpreter);
    if (buildStatus != kTfLiteOk || !nextInterpreter) {
        throw std::runtime_error("could not create TensorFlow Lite interpreter");
    }
    if (nextInterpreter->inputs().empty() ||
        nextInterpreter->outputs().empty()) {
        throw std::runtime_error(
            "model must have at least one input and one output");
    }
    nextInterpreter->SetNumThreads(backend == ExecutionBackend::CPU_FALLBACK
                                       ? 4
                                       : 2);

    TfLiteDelegate* nextGpuDelegate = nullptr;
    if (backend == ExecutionBackend::GPU_DELEGATE) {
        nextGpuDelegate =
            TfLiteGpuDelegateV2Create(nullptr);
        if (!nextGpuDelegate) {
            std::cerr << "[Scheduler Warning] TensorFlow Lite GPU delegate "
                         "is unavailable.\n";
            return false;
        }
        if (nextInterpreter->ModifyGraphWithDelegate(nextGpuDelegate) !=
            kTfLiteOk) {
            nextInterpreter.reset();
            TfLiteGpuDelegateV2Delete(nextGpuDelegate);
            throw std::runtime_error(
                "GPU delegate rejected the TensorFlow Lite model");
        }
    }

    if (nextInterpreter->AllocateTensors() != kTfLiteOk) {
        nextInterpreter.reset();
        if (nextGpuDelegate) {
            TfLiteGpuDelegateV2Delete(nextGpuDelegate);
        }
        throw std::runtime_error("could not allocate TensorFlow Lite tensors");
    }

    const TfLiteTensor* inputTensor = nextInterpreter->input_tensor(0);
    if (!inputTensor || !inputTensor->dims || inputTensor->dims->size != 4 ||
        inputTensor->type != kTfLiteFloat32) {
        nextInterpreter.reset();
        if (nextGpuDelegate) {
            TfLiteGpuDelegateV2Delete(nextGpuDelegate);
        }
        throw std::runtime_error(
            "model input must be a four-dimensional float32 tensor");
    }

    const int* dims = inputTensor->dims->data;
    const bool isNhwc = dims[0] == 1 && dims[1] == 224 &&
                        dims[2] == 224 && dims[3] == 3;
    const bool isNchw = dims[0] == 1 && dims[1] == 3 &&
                        dims[2] == 224 && dims[3] == 224;
    if (!isNhwc && !isNchw) {
        nextInterpreter.reset();
        if (nextGpuDelegate) {
            TfLiteGpuDelegateV2Delete(nextGpuDelegate);
        }
        throw std::runtime_error(
            "model input shape must be [1,224,224,3] or [1,3,224,224]");
    }

    interpreter_.reset();
    if (gpuDelegate_) {
        TfLiteGpuDelegateV2Delete(gpuDelegate_);
    }
    interpreter_ = std::move(nextInterpreter);
    gpuDelegate_ = nextGpuDelegate;
    inputIsNhwc_ = isNhwc;
    currentBackend_ = backend;

    if (backend == ExecutionBackend::GPU_DELEGATE) {
        std::cout << "[Scheduler] TensorFlow Lite GPU delegate attached. "
                     "Unsupported model operators may still execute on CPU; "
                     "check Android logs for delegate warnings.\n";
    }
    return true;
}

void InferenceScheduler::switchBackend(ExecutionBackend backend) {
    if (currentBackend_ == backend ||
        (backend == ExecutionBackend::GPU_DELEGATE && gpuUnavailable_)) {
        return;
    }

    if (backend == ExecutionBackend::CPU_FALLBACK) {
        std::cout << "[Daemon] Thermal mitigation active: switching to CPU.\n";
    } else {
        std::cout << "[Daemon] Thermal recovery: restoring the GPU delegate.\n";
    }

    try {
        if (!initializeSession(backend)) {
            gpuUnavailable_ = true;
        }
    } catch (const std::exception& e) {
        std::cerr << "[Scheduler Warning] Could not switch inference backend ("
                  << e.what() << "). Keeping the current session.\n";
        if (backend == ExecutionBackend::GPU_DELEGATE) {
            gpuUnavailable_ = true;
        }
    }
}

double InferenceScheduler::runInference(const std::vector<float>& input,
                                        std::vector<float>& output) {
    if (!interpreter_) {
        throw std::runtime_error("inference session is not initialized");
    }
    if (input.size() != kInputSize) {
        throw std::runtime_error(
            "input tensor must contain exactly 150528 float values");
    }

    TfLiteTensor* inputTensor = interpreter_->input_tensor(0);
    if (!inputTensor || inputTensor->bytes != kInputSize * sizeof(float)) {
        throw std::runtime_error(
            "model input tensor must contain exactly 150528 float values");
    }

    const auto start = std::chrono::steady_clock::now();
    float* modelInput = inputTensor->data.f;
    if (inputIsNhwc_) {
        for (size_t pixel = 0; pixel < 224 * 224; ++pixel) {
            modelInput[pixel * 3] = input[pixel];
            modelInput[pixel * 3 + 1] = input[224 * 224 + pixel];
            modelInput[pixel * 3 + 2] = input[2 * 224 * 224 + pixel];
        }
    } else {
        std::copy(input.begin(), input.end(), modelInput);
    }

    if (interpreter_->Invoke() != kTfLiteOk) {
        throw std::runtime_error("TensorFlow Lite inference failed");
    }

    const TfLiteTensor* outputTensor = interpreter_->output_tensor(0);
    if (!outputTensor || outputTensor->type != kTfLiteFloat32 ||
        outputTensor->bytes == 0 ||
        outputTensor->bytes % sizeof(float) != 0) {
        throw std::runtime_error(
            "model output must be a non-empty float32 tensor");
    }
    const size_t outputSize = outputTensor->bytes / sizeof(float);
    output.assign(outputTensor->data.f, outputTensor->data.f + outputSize);

    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}
