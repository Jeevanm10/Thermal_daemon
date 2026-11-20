#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include <memory>
#include <vector>

enum class ExecutionBackend {
    GPU_OR_NNAPI,
    CPU_FALLBACK
};

class InferenceScheduler {
public:
    explicit InferenceScheduler(const std::string& modelPath);
    void switchBackend(ExecutionBackend backend);
    double runInferenceLoopStep(bool enableInference = true);

private:
    bool initializeSession(ExecutionBackend backend);

    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;
    ExecutionBackend currentBackend_;
    std::string modelPath_;
    std::string inputName_;
    std::string outputName_;
    bool nnapiUnavailable_ = false;

    // Pre-allocated tensor buffers for MobileNet [1, 3, 224, 224]
    std::vector<float> inputTensorValues_;
    std::vector<int64_t> inputDims_;
    Ort::MemoryInfo memoryInfo_;
};