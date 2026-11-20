#include "InferenceScheduler.hpp"
#include <nnapi_provider_factory.h>
#include <dlfcn.h>
#include <iostream>
#include <chrono>
#include <stdexcept>

InferenceScheduler::InferenceScheduler(const std::string& modelPath)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ThermalDaemon"),
      currentBackend_(ExecutionBackend::GPU_OR_NNAPI),
      modelPath_(modelPath),
      inputDims_{1, 3, 224, 224},
      memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    bool nnapiInitialized = false;
    try {
        nnapiInitialized = initializeSession(ExecutionBackend::GPU_OR_NNAPI);
    } catch (const std::exception& e) {
        std::cerr << "[Scheduler Warning] NNAPI initialization failed (" << e.what() << ").\n";
    }

    if (!nnapiInitialized) {
        nnapiUnavailable_ = true;
        currentBackend_ = ExecutionBackend::CPU_FALLBACK;
        try {
            initializeSession(ExecutionBackend::CPU_FALLBACK);
        } catch (const std::exception& e) {
            std::cerr << "[Scheduler Warning] Could not load model at " << modelPath_
                      << " (" << e.what() << "). Running in mock inference mode.\n";
        }
    }

    // Initialize dummy input tensor (MobileNet standard input size: 1x3x224x224)
    size_t inputTensorSize = 1 * 3 * 224 * 224;
    inputTensorValues_.assign(inputTensorSize, 0.5f);
}

bool InferenceScheduler::initializeSession(ExecutionBackend backend) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(backend == ExecutionBackend::CPU_FALLBACK ? 4 : 2);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (backend == ExecutionBackend::GPU_OR_NNAPI) {
        auto appendNnapi = reinterpret_cast<decltype(&OrtSessionOptionsAppendExecutionProvider_Nnapi)>(
            dlsym(RTLD_DEFAULT, "OrtSessionOptionsAppendExecutionProvider_Nnapi"));
        if (!appendNnapi) {
            std::cerr << "[Scheduler Warning] NNAPI execution provider is unavailable; using CPU inference.\n";
            return false;
        }

        Ort::ThrowOnError(appendNnapi(options, NNAPI_FLAG_USE_FP16));
    }

    auto session = std::make_unique<Ort::Session>(env_, modelPath_.c_str(), options);
    auto inputNames = session->GetInputNames();
    auto outputNames = session->GetOutputNames();
    if (inputNames.empty() || outputNames.empty()) {
        throw std::runtime_error("model must have at least one input and one output");
    }

    inputName_ = inputNames.front();
    outputName_ = outputNames.front();
    sessionOptions_ = std::move(options);
    session_ = std::move(session);
    currentBackend_ = backend;
    if (backend == ExecutionBackend::GPU_OR_NNAPI) {
        std::cout << "[Scheduler] NNAPI enabled with FP16 relaxation; Android selects available accelerator devices.\n";
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

double InferenceScheduler::runInferenceLoopStep(bool enableInference) {
    if (!enableInference || !session_) {
        // Simulate lightweight cycle without inference overhead
        return 0.0;
    }

    auto start = std::chrono::high_resolution_clock::now();

    try {
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo_, inputTensorValues_.data(), inputTensorValues_.size(),
            inputDims_.data(), inputDims_.size()
        );

        const char* inputNames[] = {inputName_.c_str()};
        const char* outputNames[] = {outputName_.c_str()};

        auto outputTensors = session_->Run(
            Ort::RunOptions{nullptr}, 
            inputNames, &inputTensor, 1, 
            outputNames, 1
        );
    } catch (const std::exception& e) {
        std::cerr << "[Inference Warning] MobileNet inference failed: " << e.what() << "\n";
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}