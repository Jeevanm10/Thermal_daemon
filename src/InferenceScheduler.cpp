#include "InferenceScheduler.hpp"
#include <nnapi_provider_factory.h>
#include <dlfcn.h>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <utility>

InferenceScheduler::InferenceScheduler(const std::string& modelPath,
                                       bool requireNnapi,
                                       const std::string& profilePrefix)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ThermalDaemon"),
      currentBackend_(ExecutionBackend::GPU_OR_NNAPI),
      modelPath_(modelPath),
      requireNnapi_(requireNnapi),
      profilePrefix_(profilePrefix),
      inputDims_{1, 3, 224, 224},
      memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    bool nnapiInitialized = false;
    try {
        nnapiInitialized = initializeSession(ExecutionBackend::GPU_OR_NNAPI);
    } catch (const std::exception& e) {
        std::cerr << "[Scheduler Warning] NNAPI initialization failed (" << e.what() << ").\n";
    }

    if (!nnapiInitialized) {
        if (requireNnapi_) {
            throw std::runtime_error(
                "NNAPI could not initialize; accelerator benchmark cannot continue");
        }
        nnapiUnavailable_ = true;
        currentBackend_ = ExecutionBackend::CPU_FALLBACK;
        initializeSession(ExecutionBackend::CPU_FALLBACK);
    }

}

bool InferenceScheduler::initializeSession(ExecutionBackend backend) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(backend == ExecutionBackend::CPU_FALLBACK ? 4 : 2);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (!profilePrefix_.empty()) {
        const std::string sessionProfilePrefix =
            profilePrefix_ + "_" + std::to_string(sessionNumber_++);
        options.EnableProfiling(sessionProfilePrefix.c_str());
    }

    if (backend == ExecutionBackend::GPU_OR_NNAPI) {
        auto appendNnapi = reinterpret_cast<decltype(&OrtSessionOptionsAppendExecutionProvider_Nnapi)>(
            dlsym(RTLD_DEFAULT, "OrtSessionOptionsAppendExecutionProvider_Nnapi"));
        if (!appendNnapi) {
            std::cerr << "[Scheduler Warning] NNAPI execution provider is unavailable; using CPU inference.\n";
            return false;
        }

        uint32_t nnapiFlags = NNAPI_FLAG_USE_FP16;
        if (requireNnapi_) {
            nnapiFlags |= NNAPI_FLAG_CPU_DISABLED;
        }
        Ort::ThrowOnError(appendNnapi(options, nnapiFlags));
    }

    auto session = std::make_unique<Ort::Session>(env_, modelPath_.c_str(), options);
    auto inputNames = session->GetInputNames();
    auto outputNames = session->GetOutputNames();
    if (inputNames.empty() || outputNames.empty()) {
        throw std::runtime_error("model must have at least one input and one output");
    }

    inputName_ = inputNames.front();
    outputName_ = outputNames.front();
    const auto inputTypeInfo = session->GetInputTypeInfo(0);
    const auto tensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    const auto inputElementType = tensorInfo.GetElementType();
    if (inputElementType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        inputElementType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        throw std::runtime_error(
            "model input must be float32 or float16");
    }

    const size_t inputTensorSize = 1 * 3 * 224 * 224;
    if (inputElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        inputTensorFp16Values_.assign(
            inputTensorSize, Ort::Float16_t(0.0f));
        inputTensorValues_.clear();
    } else {
        inputTensorValues_.resize(inputTensorSize);
        inputTensorFp16Values_.clear();
    }

    session_ = std::move(session);
    inputElementType_ = inputElementType;
    currentBackend_ = backend;
    if (backend == ExecutionBackend::GPU_OR_NNAPI) {
        std::cout << "[Scheduler] NNAPI execution provider enabled"
                  << (requireNnapi_
                          ? " with NNAPI CPU devices disabled where supported"
                          : "")
                  << ". NNAPI may use a GPU or NPU; this does not identify a specific device.\n";
    }
    return true;
}

void InferenceScheduler::switchBackend(ExecutionBackend backend) {
    if (currentBackend_ == backend || (backend == ExecutionBackend::GPU_OR_NNAPI && nnapiUnavailable_)) return;

    if (backend == ExecutionBackend::CPU_FALLBACK) {
        std::cout << "[Daemon] Thermal mitigation active: Shifting MobileNet workload to CPU fallback.\n";
    } else {
        std::cout << "[Daemon] Thermal recovery: Restoring MobileNet workload to NNAPI (GPU/accelerator when available).\n";
    }

    try {
        if (!initializeSession(backend)) {
            nnapiUnavailable_ = true;
            if (!session_) {
                initializeSession(ExecutionBackend::CPU_FALLBACK);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[Scheduler Warning] Could not switch inference backend ("
                  << e.what() << "). Keeping the current session.\n";
        if (backend == ExecutionBackend::GPU_OR_NNAPI) {
            nnapiUnavailable_ = true;
        }
    }
}

double InferenceScheduler::runInference(const std::vector<float>& input,
                                        std::vector<float>& output) {
    const size_t expectedSize = 1 * 3 * 224 * 224;
    if (!session_) {
        throw std::runtime_error("inference session is not initialized");
    }
    if (input.size() != expectedSize) {
        throw std::runtime_error("input tensor must contain exactly 150528 float values");
    }

    const auto start = std::chrono::steady_clock::now();
    Ort::Value inputTensor{nullptr};
    if (inputElementType_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        for (size_t i = 0; i < input.size(); ++i) {
            inputTensorFp16Values_[i] = Ort::Float16_t(input[i]);
        }
        inputTensor = Ort::Value::CreateTensor<Ort::Float16_t>(
            memoryInfo_, inputTensorFp16Values_.data(),
            inputTensorFp16Values_.size(), inputDims_.data(),
            inputDims_.size());
    } else {
        inputTensorValues_ = input;
        inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo_, inputTensorValues_.data(), inputTensorValues_.size(),
            inputDims_.data(), inputDims_.size());
    }

    const char* inputNames[] = {inputName_.c_str()};
    const char* outputNames[] = {outputName_.c_str()};
    auto outputTensors = session_->Run(
        Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);
    if (outputTensors.empty()) {
        throw std::runtime_error("model returned no output tensor");
    }

    const auto outputInfo =
        outputTensors.front().GetTensorTypeAndShapeInfo();
    const size_t outputSize = outputInfo.GetElementCount();
    if (outputSize == 0) {
        throw std::runtime_error("model returned an empty output tensor");
    }

    switch (outputInfo.GetElementType()) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            const float* outputData =
                outputTensors.front().GetTensorData<float>();
            output.assign(outputData, outputData + outputSize);
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
            const Ort::Float16_t* outputData =
                outputTensors.front().GetTensorData<Ort::Float16_t>();
            output.resize(outputSize);
            for (size_t i = 0; i < outputSize; ++i) {
                output[i] = outputData[i].ToFloat();
            }
            break;
        }
        default:
            throw std::runtime_error(
                "unsupported model output tensor type " +
                std::to_string(static_cast<int>(outputInfo.GetElementType())) +
                "; classification scores must be float32 or float16");
    }

    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string InferenceScheduler::endProfiling() {
    if (!session_ || profilePrefix_.empty()) {
        return "";
    }
    Ort::AllocatorWithDefaultOptions allocator;
    auto profilePath = session_->EndProfilingAllocated(allocator);
    return profilePath.get();
}