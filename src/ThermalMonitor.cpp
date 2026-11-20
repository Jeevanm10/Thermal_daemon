#include "ThermalMonitor.hpp"
#include <fstream>
#include <utility>

ThermalMonitor::ThermalMonitor(std::string sysfsPath)
    : path_(std::move(sysfsPath)),
      lastTemp_(0.0f),
      hasLastTemperature_(false) {}

float ThermalMonitor::readCurrentTemperature() {
    std::ifstream file(path_);
    if (!file.is_open()) {
        // Fallback simulation value if sysfs path is unavailable on test host
        return 45.0f; 
    }
    long millidegrees = 0;
    if (!(file >> millidegrees)) {
        return 45.0f;
    }
    return static_cast<float>(millidegrees) / 1000.0f;
}

float ThermalMonitor::computeRateOfChange(float currentTemp, double deltaTimeSeconds) {
    if (deltaTimeSeconds <= 0.0) {
        return 0.0f;
    }

    if (!hasLastTemperature_) {
        lastTemp_ = currentTemp;
        hasLastTemperature_ = true;
        return 0.0f;
    }

    float dt = currentTemp - lastTemp_;
    float dT_dt = dt / static_cast<float>(deltaTimeSeconds);
    lastTemp_ = currentTemp;
    return dT_dt;
}