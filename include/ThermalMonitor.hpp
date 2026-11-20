#pragma once
#include <string>

class ThermalMonitor {
public:
    explicit ThermalMonitor(std::string sysfsPath);
    float readCurrentTemperature();
    float computeRateOfChange(float currentTemp, double deltaTimeSeconds);

private:
    std::string path_;
    float lastTemp_;
    bool hasLastTemperature_;
};