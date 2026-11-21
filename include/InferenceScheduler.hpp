#pragma once

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

#include <memory>
#include <string>
#include <vector>

enum class ExecutionBackend {
    GPU_DELEGATE,
    CPU_FALLBACK
};

class InferenceScheduler {
public:
    explicit InferenceScheduler(const std::string& modelPath);
    ~InferenceScheduler();

    InferenceScheduler(const InferenceScheduler&) = delete;
    InferenceScheduler& operator=(const InferenceScheduler&) = delete;

    void switchBackend(ExecutionBackend backend);
    double runInference(const std::vector<float>& input,
                        std::vector<float>& output);
    ExecutionBackend currentBackend() const { return currentBackend_; }

private:
    bool initializeSession(ExecutionBackend backend);

    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    TfLiteDelegate* gpuDelegate_ = nullptr;
    ExecutionBackend currentBackend_ = ExecutionBackend::CPU_FALLBACK;
    bool gpuUnavailable_ = false;
    bool inputIsNhwc_ = false;
};
