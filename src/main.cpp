#include "Config.hpp"
#include "ThermalMonitor.hpp"
#include "InferenceScheduler.hpp"

#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>

namespace {
volatile std::sig_atomic_t g_running = true;

void signalHandler(int) {
    g_running = false;
}
}

int main() {
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    std::cout << "[Daemon] Starting thermal-aware daemon...\n";

    try {
        ThermalMonitor monitor(ThermalConfig::THERMAL_ZONE_PATH);
        InferenceScheduler scheduler(
            "/data/local/tmp/mobilenetv2.onnx");
        auto lastLoopTime = std::chrono::steady_clock::now();

        while (g_running) {
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
                scheduler.switchBackend(ExecutionBackend::GPU_OR_NNAPI);
            }

            scheduler.runInferenceLoopStep();
            std::this_thread::sleep_for(
                std::chrono::milliseconds(
                    ThermalConfig::SAMPLING_INTERVAL_MS));
        }
    } catch (const std::exception& e) {
        std::cerr << "[Monitor Error] " << e.what() << "\n";
        return 1;
    }

    std::cout << "[Monitor] Terminated gracefully.\n";
    return 0;
}
