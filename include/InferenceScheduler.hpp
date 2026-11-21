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
    explicit InferenceScheduler(const std::string& modelPath,
                                bool requireNnapi = false,
                                const std::string& profilePrefix = "");
    void switchBackend(ExecutionBackend backend);
    double runInference(const std::vector<float>& input,
                        std::vector<float>& output);
    std::string endProfiling();
    ExecutionBackend currentBackend() const { return currentBackend_; }

private:
    bool initializeSession(ExecutionBackend backend);

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    ExecutionBackend currentBackend_;
    std::string modelPath_;
    std::string inputName_;
    std::string outputName_;
    bool nnapiUnavailable_ = false;
    bool requireNnapi_;
    std::string profilePrefix_;
    size_t sessionNumber_ = 0;
    ONNXTensorElementDataType inputElementType_ =
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

    // Input buffers for the model tensor [1, 3, 224, 224].
    std::vector<float> inputTensorValues_;
    std::vector<Ort::Float16_t> inputTensorFp16Values_;
    std::vector<int64_t> inputDims_;
    Ort::MemoryInfo memoryInfo_;
};