#ifndef DIAG_LOGGER_HPP
#define DIAG_LOGGER_HPP

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

// Lightweight CSV diagnostics for TLCacheMAB TEST/TRAIN runs.
// Produces periodic snapshots and discrete events.
//   <prefix>_timeseries.csv - miss/weight/arm snapshots
//   <prefix>_events.csv  - policy inject, start, shutdown
class DiagLogger {
public:
    DiagLogger() = default;
    ~DiagLogger();

    void init(bool enable,
              const std::string& prefix,
              uint64_t interval,
              uint8_t arm_count);

    bool enabled() const { return enabled_; }
    uint64_t interval() const { return interval_; }

    void log_timeseries(uint64_t seq,
                        const char* mode_name,
                        uint8_t last_arm,
                        double ema_miss,
                        double window_miss,
                        double cum_miss,
                        double miss_slope,
                        const double* weights,
                        const double* probs,
                        const uint64_t* arm_evicts,
                        const uint64_t* arm_selects);

    void log_event(uint64_t seq,
                   const char* event_type,
                   const char* detail,
                   double ema_miss,
                   double cum_miss,
                   const double* weights);

    void flush();

private:
    bool enabled_ = false;
    uint64_t interval_ = 10000;
    uint8_t arm_count_ = 4;
    std::string prefix_;
    std::ofstream ts_file_;
    std::ofstream ev_file_;
    bool ts_header_written_ = false;
    bool ev_header_written_ = false;

    void ensure_ts_header();
    void ensure_ev_header();
    static bool ensure_parent_dir(const std::string& filepath);
};

#endif // DIAG_LOGGER_HPP
