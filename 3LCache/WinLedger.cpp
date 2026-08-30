#include "WinLedger.hpp"

#include <filesystem>
#include <iostream>

static const char* kFeatureNames[WinLedger::N_FEATURES] = {
    "write_ratio",
    "avg_request_size",
    "size_variance",
    "singleton_ratio",
    "popularity_skewness",
    "avg_frequency",
    "object_diversity",
    "avg_reuse_distance",
    "sequentiality_ratio",
    "out_cache_hit_rate",
    "request_rate",
    "burstiness_index",
    "arrival_time_variance",
    "working_set_byte_delta",
    "scan_ratio"
};

bool WinLedger::ensure_parent_dir(const std::string& filepath) {
    if (filepath.empty()) return false;
    std::filesystem::path parent = std::filesystem::path(filepath).parent_path();
    if (parent.empty()) return true;
    std::error_code ec;
    if (std::filesystem::exists(parent, ec)) return true;
    return std::filesystem::create_directories(parent, ec);
}

WinLedger::~WinLedger() {
    flush();
    if (file_.is_open()) file_.close();
}

void WinLedger::init(bool enable,
                     const std::string& prefix,
                     uint64_t window,
                     uint8_t arm_count,
                     const std::string& run_tag) {
    enabled_ = enable;
    prefix_ = prefix.empty() ? "mab_win" : prefix;
    window_ = (window == 0) ? 100000 : window;
    arm_count_ = (arm_count == 0) ? 4 : arm_count;
    run_tag_ = run_tag.empty() ? "run" : run_tag;

    if (!enabled_) return;

    const std::string path = prefix_ + "_windows.csv";
    ensure_parent_dir(path);
    file_.open(path, std::ios::trunc);
    if (!file_.is_open()) {
        std::cerr << "[WinLedger] failed to open " << path << "\n";
        enabled_ = false;
        return;
    }

    header_written_ = false;
    ensure_header();
    flush();
}

void WinLedger::ensure_header() {
    if (header_written_ || !file_.is_open()) return;
    file_ << "run,win_idx,seq_end,reqs,misses,bytes_req,bytes_miss,omr,bmr,"
             "cum_omr,cum_bmr,dom_arm,matrix_rows";
    for (uint8_t i = 0; i < arm_count_; ++i) file_ << ",w" << (int)i;
    for (int i = 0; i < N_FEATURES; ++i) file_ << "," << kFeatureNames[i];
    // Appended after the legacy columns so existing column indices stay valid.
    file_ << ",arm_phase,committed_arm,regime_id,injects";
    for (int i = 0; i < N_FEATURES; ++i) file_ << ",min_" << kFeatureNames[i];
    for (int i = 0; i < N_FEATURES; ++i) file_ << ",max_" << kFeatureNames[i];
    file_ << ",shadow_misses,shadow_bytes_miss,shadow_omr,shadow_bmr,"
             "delta_omr,delta_bmr,win_vs_shadow";
    file_ << "\n";
    header_written_ = true;
}

void WinLedger::log_window(uint64_t win_idx,
                           uint64_t seq_end,
                           uint64_t reqs,
                           uint64_t misses,
                           uint64_t bytes_req,
                           uint64_t bytes_miss,
                           uint64_t cum_reqs,
                           uint64_t cum_misses,
                           uint64_t cum_bytes_req,
                           uint64_t cum_bytes_miss,
                           int dominant_arm,
                           size_t matrix_rows,
                           const double* weights,
                           const std::vector<double>& features,
                           int arm_phase,
                           int committed_arm,
                           uint64_t regime_id,
                           uint64_t injects,
                           const std::vector<double>& feat_min,
                           const std::vector<double>& feat_max,
                           uint64_t shadow_misses,
                           uint64_t shadow_bytes_miss) {
    if (!enabled_ || !file_.is_open()) return;
    ensure_header();

    const double omr = (reqs > 0) ? (double)misses / (double)reqs : 0.0;
    const double bmr = (bytes_req > 0) ? (double)bytes_miss / (double)bytes_req : 0.0;
    const double cum_omr = (cum_reqs > 0) ? (double)cum_misses / (double)cum_reqs : 0.0;
    const double cum_bmr = (cum_bytes_req > 0)
                               ? (double)cum_bytes_miss / (double)cum_bytes_req
                               : 0.0;

    file_ << run_tag_ << ","
          << win_idx << ","
          << seq_end << ","
          << reqs << ","
          << misses << ","
          << bytes_req << ","
          << bytes_miss << ","
          << omr << ","
          << bmr << ","
          << cum_omr << ","
          << cum_bmr << ","
          << dominant_arm << ","
          << matrix_rows;

    for (uint8_t i = 0; i < arm_count_; ++i) {
        file_ << "," << (weights ? weights[i] : 0.0);
    }
    for (int i = 0; i < N_FEATURES; ++i) {
        double v = (i < (int)features.size()) ? features[i] : 0.0;
        file_ << "," << v;
    }

    file_ << "," << arm_phase
          << "," << committed_arm
          << "," << regime_id
          << "," << injects;
    for (int i = 0; i < N_FEATURES; ++i) {
        file_ << "," << ((i < (int)feat_min.size()) ? feat_min[i] : 0.0);
    }
    for (int i = 0; i < N_FEATURES; ++i) {
        file_ << "," << ((i < (int)feat_max.size()) ? feat_max[i] : 0.0);
    }
    const double shadow_omr =
        (reqs > 0) ? (double)shadow_misses / (double)reqs : 0.0;
    const double shadow_bmr =
        (bytes_req > 0) ? (double)shadow_bytes_miss / (double)bytes_req : 0.0;
    const double delta_omr = shadow_omr - omr;
    const double delta_bmr = shadow_bmr - bmr;
    file_ << "," << shadow_misses
          << "," << shadow_bytes_miss
          << "," << shadow_omr
          << "," << shadow_bmr
          << "," << delta_omr
          << "," << delta_bmr
          << "," << ((delta_bmr > 0.0) ? 1 : 0);
    file_ << "\n";

    if ((win_idx % 20) == 0) file_.flush();
}

void WinLedger::flush() {
    if (file_.is_open()) file_.flush();
}
