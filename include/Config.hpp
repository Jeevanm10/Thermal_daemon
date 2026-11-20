#pragma once
#include <string>

namespace ThermalConfig {
    constexpr int SAMPLING_INTERVAL_MS = 50;
    inline const std::string THERMAL_ZONE_PATH =
        "/sys/class/thermal/thermal_zone1/temp";
    
    // Thresholds in Celsius
    constexpr float TEMP_WARNING_THRESHOLD_C = 65.0f;
    constexpr float TEMP_CRITICAL_THRESHOLD_C = 75.0f;
    constexpr float HYSTERESIS_RECOVERY_DELTA_C = 5.0f; // Recovery window
    
    // Rate of change threshold (dT/dt) indicating a thermal spike (°C/s)
    constexpr float DT_DT_SPIKE_THRESHOLD = 1.5f; 
}