#include "TrainLogger.hpp"
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <cmath>

bool TrainLogger::ensure_parent_dir(const std::string& filepath) {
    if (filepath.empty()) return false;
    std::filesystem::path parent = std::filesystem::path(filepath).parent_path();
    if (parent.empty()) return true;
    std::error_code ec;
    if (std::filesystem::exists(parent, ec)) return true;
    return std::filesystem::create_directories(parent, ec);
}

TrainLogger::~TrainLogger() {
    flush();
    if (progress_file_.is_open()) progress_file_.close();
    if (learning_file_.is_open()) learning_file_.close();
}

void TrainLogger::init(bool enable,
                       const std::string& prefix,
                       uint64_t interval,
                       uint8_t arm_count) {
    enabled_ = enable;
    prefix_ = prefix.empty() ? "mab_train" : prefix;
    interval_ = (interval == 0) ? 10000 : interval;
    arm_count_ = arm_count == 0 ? 4 : arm_count;

    if (!enabled_) return;

    const std::string progress_path = prefix_ + "_progress.csv";
    const std::string learning_path = prefix_ + "_learning.csv";
    ensure_parent_dir(progress_path);
    ensure_parent_dir(learning_path);

    // Progress is the large per-run time-series: truncate each process.
    progress_file_.open(progress_path, std::ios::trunc);

    // Learning is the compact store/skip ledger (one row per event). Each
    // trace runs in its own cachesim process, so append here to keep the whole
    // training history in one file; only write the header when it is new.
    std::error_code ec;
    bool learning_exists = std::filesystem::exists(learning_path, ec) &&
                           std::filesystem::file_size(learning_path, ec) > 0;
    learning_file_.open(learning_path, std::ios::app);

    if (!progress_file_.is_open() || !learning_file_.is_open()) {
        std::cerr << "[TrainLogger] failed to open files under prefix=" << prefix_ << "\n";
        enabled_ = false;
        return;
    }

    progress_header_ = false;
    learning_header_ = learning_exists; // skip header if appending to existing file
    ensure_progress_header();
    ensure_learning_header();
    flush();
}

void TrainLogger::ensure_progress_header() {
    if (progress_header_ || !progress_file_.is_open()) return;
    progress_file_ << "seq,matrix_rows,phase_miss,cum_miss,shift,shift_threshold,weight_delta,dominant_arm,max_min_ratio";
    for (uint8_t i = 0; i < arm_count_; ++i) progress_file_ << ",w" << (int)i;
    progress_file_ << "\n";
    progress_header_ = true;
}

void TrainLogger::ensure_learning_header() {
    if (learning_header_ || !learning_file_.is_open()) return;
    learning_file_ << "seq,action,matrix_rows,regime_delta_vs_shadow,dist_to_nearest,dominant_arm,max_min_ratio";
    for (uint8_t i = 0; i < arm_count_; ++i) learning_file_ << ",w" << (int)i;
    learning_file_ << "\n";
    learning_header_ = true;
}

static int dominant_arm_of(const std::vector<double>& weights) {
    if (weights.empty()) return -1;
    size_t best = 0;
    for (size_t i = 1; i < weights.size(); ++i) {
        if (weights[i] > weights[best]) best = i;
    }
    return (int)best;
}

static double max_min_ratio_of(const std::vector<double>& weights) {
    if (weights.empty()) return 0.0;
    double mx = weights[0], mn = weights[0];
    for (double w : weights) {
        if (w > mx) mx = w;
        if (w < mn) mn = w;
    }
    if (mn <= 0.0) return 1e300;
    return mx / mn;
}

void TrainLogger::log_progress(uint64_t seq,
                               size_t matrix_rows,
                               double phase_miss,
                               double cum_miss,
                               double shift,
                               double shift_threshold,
                               double weight_delta,
                               const std::vector<double>& weights) {
    if (!enabled_ || !progress_file_.is_open()) return;
    ensure_progress_header();

    progress_file_ << seq << ","
                   << matrix_rows << ","
                   << phase_miss << ","
                   << cum_miss << ","
                   << shift << ","
                   << shift_threshold << ","
                   << weight_delta << ","
                   << dominant_arm_of(weights) << ","
                   << max_min_ratio_of(weights);
    for (uint8_t i = 0; i < arm_count_; ++i) {
        double w = (i < weights.size()) ? weights[i] : 0.0;
        progress_file_ << "," << w;
    }
    progress_file_ << "\n";
    if ((seq / interval_) % 10 == 0) progress_file_.flush();
}

void TrainLogger::log_learning(uint64_t seq,
                               const char* action,
                               size_t matrix_rows,
                               double phase_miss,
                               double dist_to_nearest,
                               int dominant_arm,
                               const std::vector<double>& weights) {
    if (!enabled_ || !learning_file_.is_open()) return;
    ensure_learning_header();

    int dom = (dominant_arm >= 0) ? dominant_arm : dominant_arm_of(weights);
    learning_file_ << seq << ","
                   << (action ? action : "event") << ","
                   << matrix_rows << ","
                   << phase_miss << ","
                   << dist_to_nearest << ","
                   << dom << ","
                   << max_min_ratio_of(weights);

    for (uint8_t i = 0; i < arm_count_; ++i) {
        double w = (i < weights.size()) ? weights[i] : 0.0;
        learning_file_ << "," << w;
    }
    learning_file_ << "\n";
    learning_file_.flush();
}

void TrainLogger::flush() {
    if (progress_file_.is_open()) progress_file_.flush();
    if (learning_file_.is_open()) learning_file_.flush();
}
