#include "Config.hpp"
#include "ThermalMonitor.hpp"
#include "InferenceScheduler.hpp"

#include <chrono>
#include <algorithm>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
volatile std::sig_atomic_t g_running = true;

void signalHandler(int) {
    g_running = false;
}

struct Prediction {
    std::string tensorName;
    size_t top1;
    float score;
    double latencyMs;
    std::string backend;
};

std::vector<std::filesystem::path> findTensorFiles(
    const std::string& inputDirectory) {
    std::vector<std::filesystem::path> tensorFiles;
    for (const auto& entry : std::filesystem::directory_iterator(inputDirectory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".tensor") {
            tensorFiles.push_back(entry.path());
        }
    }
    std::sort(tensorFiles.begin(), tensorFiles.end());
    return tensorFiles;
}

std::vector<float> loadTensor(const std::filesystem::path& tensorFile) {
    constexpr size_t inputSize = 3 * 224 * 224;
    std::ifstream inputFile(tensorFile, std::ios::binary);
    if (!inputFile) {
        throw std::runtime_error("could not open tensor: " + tensorFile.string());
    }
    std::vector<float> input(inputSize);
    inputFile.read(reinterpret_cast<char*>(input.data()),
                   static_cast<std::streamsize>(input.size() * sizeof(float)));
    if (!inputFile ||
        inputFile.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error(
            "tensor must contain exactly 150528 float32 values: " +
            tensorFile.string());
    }
    return input;
}

void writeCsvField(std::ostream& output, const std::string& value) {
    output << '"';
    for (const char character : value) {
        if (character == '"') {
            output << '"';
        }
        output << character;
    }
    output << '"';
}

void writePredictionReport(const std::string& reportPath,
                           const std::vector<Prediction>& predictions) {
    std::ofstream report(reportPath, std::ios::trunc);
    if (!report) {
        throw std::runtime_error("could not write prediction report: " + reportPath);
    }
    report << "tensor,top1,score,latency_ms,backend\n";
    for (const auto& prediction : predictions) {
        writeCsvField(report, prediction.tensorName);
        report << ',' << prediction.top1 << ',' << std::setprecision(9)
               << prediction.score << ',' << prediction.latencyMs << ',';
        writeCsvField(report, prediction.backend);
        report << '\n';
    }
    if (!report) {
        throw std::runtime_error("failed while writing prediction report: " +
                                 reportPath);
    }
}

void runBenchmark(const std::string& inputDirectory) {
    const auto tensorFiles = findTensorFiles(inputDirectory);
    if (tensorFiles.empty()) {
        throw std::runtime_error(
            "no .tensor inputs found; run tools/prepare_images.py first");
    }

    InferenceScheduler scheduler("/data/local/tmp/mobilenetv2.tflite");
    const std::string reportPath =
        "/data/local/tmp/thermal_benchmark_results.csv";
    std::ofstream report(reportPath, std::ios::trunc);
    if (!report) {
        throw std::runtime_error("could not write prediction report: " +
                                 reportPath);
    }
    report << "pass,tensor,top1,score,latency_ms\n";

    const auto runPass = [&](const std::string& passName,
                             ExecutionBackend expectedBackend) {
        if (scheduler.currentBackend() != expectedBackend) {
            throw std::runtime_error(passName + " backend switch did not take effect");
        }
        std::cout << "[Benchmark] Starting " << passName << " on "
                  << (expectedBackend == ExecutionBackend::CPU_FALLBACK
                          ? "CPU"
                          : "TensorFlow Lite GPU delegate")
                  << ".\n";
        for (const auto& tensorFile : tensorFiles) {
            const auto input = loadTensor(tensorFile);
            std::vector<float> output;
            const double durationMs = scheduler.runInference(input, output);
            if (output.empty()) {
                throw std::runtime_error("model returned an empty output tensor");
            }
            const auto best = std::max_element(output.begin(), output.end());
            const size_t classIndex =
                static_cast<size_t>(std::distance(output.begin(), best));
            const std::string tensorName = tensorFile.filename().string();
            std::cout << "[Benchmark] " << tensorName
                      << " top1=" << classIndex << " score=" << *best
                      << " latency_ms=" << durationMs << "\n";
            writeCsvField(report, passName);
            report << ',';
            writeCsvField(report, tensorName);
            report << ',' << classIndex << ',' << std::setprecision(9) << *best
                   << ',' << durationMs << '\n';
            report.flush();
            if (!report) {
                throw std::runtime_error("failed while writing prediction report: " +
                                         reportPath);
            }
        }

    };

    runPass("GPU delegate pass", ExecutionBackend::GPU_DELEGATE);
    scheduler.switchBackend(ExecutionBackend::CPU_FALLBACK);
    runPass("CPU fallback pass", ExecutionBackend::CPU_FALLBACK);
    scheduler.switchBackend(ExecutionBackend::GPU_DELEGATE);
    runPass("GPU delegate recovery pass", ExecutionBackend::GPU_DELEGATE);
    std::cout << "[Benchmark] GPU delegate -> CPU -> GPU delegate switching "
                 "verified. Delegate attachment does not prove that every "
                 "model operation ran on the GPU.\n";
    std::cout << "[Benchmark] Predictions: " << reportPath << "\n";
}
}

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--benchmark") {
        try {
            runBenchmark(argv[2]);
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "[Benchmark Error] " << e.what() << "\n";
            return 1;
        }
    }
    if (argc != 1) {
        std::cerr << "Usage: thermal_daemon [--benchmark <tensor-directory>]\n";
        return 2;
    }

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    std::cout << "[Daemon] Starting thermal-aware daemon...\n";

    try {
        const auto tensorFiles =
            findTensorFiles("/data/local/tmp/thermal_inputs");
        if (tensorFiles.empty()) {
            throw std::runtime_error(
                "no image tensors found in /data/local/tmp/thermal_inputs; "
                "prepare and push them with tools/prepare_images.py");
        }
        ThermalMonitor monitor(ThermalConfig::THERMAL_ZONE_PATH);
        InferenceScheduler scheduler("/data/local/tmp/mobilenetv2.tflite");
        std::vector<Prediction> latestPredictions(tensorFiles.size());
        size_t nextTensor = 0;
        const std::string reportPath =
            "/data/local/tmp/thermal_results.csv";
        const auto runStart = std::chrono::steady_clock::now();
        const auto runDeadline =
            runStart + std::chrono::minutes(ThermalConfig::MAX_RUNTIME_MINUTES);
        auto lastLoopTime = runStart;
        size_t completedCycles = 0;

        while (g_running && std::chrono::steady_clock::now() < runDeadline) {
            const float currentTemp = monitor.readCurrentTemperature();
            const auto currentTime = std::chrono::steady_clock::now();
            const std::chrono::duration<double> elapsed =
                currentTime - lastLoopTime;
            lastLoopTime = currentTime;

            const float dT_dt =
                monitor.computeRateOfChange(currentTemp, elapsed.count());

            if (currentTemp >= ThermalConfig::TEMP_CRITICAL_THRESHOLD_C ||
                (dT_dt >= ThermalConfig::DT_DT_SPIKE_THRESHOLD &&
                 currentTemp >= ThermalConfig::TEMP_WARNING_THRESHOLD_C)) {
                scheduler.switchBackend(ExecutionBackend::CPU_FALLBACK);
            } else if (currentTemp <
                       (ThermalConfig::TEMP_WARNING_THRESHOLD_C -
                        ThermalConfig::HYSTERESIS_RECOVERY_DELTA_C)) {
                scheduler.switchBackend(ExecutionBackend::GPU_DELEGATE);
            }

            const auto& tensorFile = tensorFiles[nextTensor];
            const auto input = loadTensor(tensorFile);
            std::vector<float> output;
            const double durationMs = scheduler.runInference(input, output);
            if (output.empty()) {
                throw std::runtime_error("model returned an empty output tensor");
            }
            const auto best = std::max_element(output.begin(), output.end());
            const size_t classIndex =
                static_cast<size_t>(std::distance(output.begin(), best));
            const std::string backend =
                scheduler.currentBackend() == ExecutionBackend::CPU_FALLBACK
                    ? "CPU"
                    : "GPU delegate";
            latestPredictions[nextTensor] = {
                tensorFile.filename().string(), classIndex, *best, durationMs,
                backend};
            std::cout << "[Inference] " << tensorFile.filename().string()
                      << " top1=" << classIndex << " score=" << *best
                      << " latency_ms=" << durationMs
                      << " backend=" << backend << "\n";

            nextTensor = (nextTensor + 1) % tensorFiles.size();
            if (nextTensor == 0) {
                ++completedCycles;
                writePredictionReport(reportPath, latestPredictions);
                std::cout << "[Inference] Completed cycle count: "
                          << completedCycles << "; updated predictions: "
                          << reportPath << "\n";
            }
            const auto nextLoop = std::chrono::steady_clock::now() +
                                  std::chrono::milliseconds(
                                      ThermalConfig::SAMPLING_INTERVAL_MS);
            std::this_thread::sleep_until(std::min(nextLoop, runDeadline));
        }
        std::cout << "[Daemon] Stopped after "
                  << ThermalConfig::MAX_RUNTIME_MINUTES
                  << " minute limit or shutdown request. Completed cycles: "
                  << completedCycles << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[Monitor Error] " << e.what() << "\n";
        return 1;
    }

    std::cout << "[Monitor] Terminated gracefully.\n";
    return 0;
}
