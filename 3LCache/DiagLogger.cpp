#include "DiagLogger.hpp"
#include <filesystem>
#include <iostream>
#include <sstream>

bool DiagLogger::ensure_parent_dir(const std::string& filepath) {
    if (filepath.empty()) return false;
    std::filesystem::path parent = std::filesystem::path(filepath).parent_path();
    if (parent.empty()) return true;
    std::error_code ec;
    if (std::filesystem::exists(parent, ec)) return true;
    return std::filesystem::create_directories(parent, ec);
}

DiagLogger::~DiagLogger() {
    flush();
    if (ts_file_.is_open()) ts_file_.close();
    if (ev_file_.is_open()) ev_file_.close();
}

void DiagLogger::init(bool enable,
                      const std::string& prefix,
                      uint64_t interval,
                      uint8_t arm_count) {
    enabled_ = enable;
    prefix_ = prefix.empty() ? "mab_diag" : prefix;
    interval_ = (interval == 0) ? 10000 : interval;
    arm_count_ = arm_count == 0 ? 4 : arm_count;

    if (!enabled_) return;

    const std::string ts_path = prefix_ + "_timeseries.csv";
    const std::string ev_path = prefix_ + "_events.csv";
    ensure_parent_dir(ts_path);
    ensure_parent_dir(ev_path);

    ts_file_.open(ts_path, std::ios::trunc);
    ev_file_.open(ev_path, std::ios::trunc);
    if (!ts_file_.is_open() || !ev_file_.is_open()) {
        std::cerr << "[DiagLogger] failed to open files under prefix=" << prefix_ << "\n";
        enabled_ = false;
        return;
    }

    ts_header_written_ = false;
    ev_header_written_ = false;
    ensure_ts_header();
    ensure_ev_header();
    flush();
}

void DiagLogger::ensure_ts_header() {
    if (ts_header_written_ || !ts_file_.is_open()) return;
    ts_file_ << "seq,mode,last_arm,ema_miss,window_miss,cum_miss,miss_slope";
    for (uint8_t i = 0; i < arm_count_; ++i) ts_file_ << ",w" << (int)i;
    for (uint8_t i = 0; i < arm_count_; ++i) ts_file_ << ",p" << (int)i;
    for (uint8_t i = 0; i < arm_count_; ++i) ts_file_ << ",evict" << (int)i;
    for (uint8_t i = 0; i < arm_count_; ++i) ts_file_ << ",select" << (int)i;
    ts_file_ << "\n";
    ts_header_written_ = true;
}

void DiagLogger::ensure_ev_header() {
    if (ev_header_written_ || !ev_file_.is_open()) return;
    ev_file_ << "seq,event,detail,ema_miss,cum_miss";
    for (uint8_t i = 0; i < arm_count_; ++i) ev_file_ << ",w" << (int)i;
    ev_file_ << "\n";
    ev_header_written_ = true;
}

void DiagLogger::log_timeseries(uint64_t seq,
                                const char* mode_name,
                                uint8_t last_arm,
                                double ema_miss,
                                double window_miss,
                                double cum_miss,
                                double miss_slope,
                                const double* weights,
                                const double* probs,
                                const uint64_t* arm_evicts,
                                const uint64_t* arm_selects) {
    if (!enabled_ || !ts_file_.is_open()) return;
    ensure_ts_header();

    ts_file_ << seq << ","
             << (mode_name ? mode_name : "unknown") << ","
             << (int)last_arm << ","
             << ema_miss << ","
             << window_miss << ","
             << cum_miss << ","
             << miss_slope;
    for (uint8_t i = 0; i < arm_count_; ++i) ts_file_ << "," << (weights ? weights[i] : 0.0);
    for (uint8_t i = 0; i < arm_count_; ++i) ts_file_ << "," << (probs ? probs[i] : 0.0);
    for (uint8_t i = 0; i < arm_count_; ++i) ts_file_ << "," << (arm_evicts ? arm_evicts[i] : 0ULL);
    for (uint8_t i = 0; i < arm_count_; ++i) ts_file_ << "," << (arm_selects ? arm_selects[i] : 0ULL);
    ts_file_ << "\n";
    if ((seq / interval_) % 10 == 0) ts_file_.flush();
}

void DiagLogger::log_event(uint64_t seq,
                           const char* event_type,
                           const char* detail,
                           double ema_miss,
                           double cum_miss,
                           const double* weights) {
    if (!enabled_ || !ev_file_.is_open()) return;
    ensure_ev_header();

    // Escape commas in detail lightly by replacing with ';'.
    std::string d = detail ? detail : "";
    for (char& c : d) {
        if (c == ',' || c == '\n') c = ';';
    }

    ev_file_ << seq << ","
             << (event_type ? event_type : "event") << ","
             << d << ","
             << ema_miss << ","
             << cum_miss;
    for (uint8_t i = 0; i < arm_count_; ++i) ev_file_ << "," << (weights ? weights[i] : 0.0);
    ev_file_ << "\n";
    ev_file_.flush();
}

void DiagLogger::flush() {
    if (ts_file_.is_open()) ts_file_.flush();
    if (ev_file_.is_open()) ev_file_.flush();
}
