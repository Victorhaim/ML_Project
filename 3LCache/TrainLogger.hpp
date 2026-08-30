#ifndef TRAIN_LOGGER_HPP
#define TRAIN_LOGGER_HPP

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

// Training-only diagnostics (separate from TEST DiagLogger).
// Records periodic training health and discrete matrix events.
//   <prefix>_progress.csv  - periodic miss/weight/matrix snapshots
//   <prefix>_learning.csv  - matrix upsert / phase events
class TrainLogger {
public:
    TrainLogger() = default;
    ~TrainLogger();

    void init(bool enable,
              const std::string& prefix,
              uint64_t interval,
              uint8_t arm_count);

    bool enabled() const { return enabled_; }
    uint64_t interval() const { return interval_; }

    void log_progress(uint64_t seq,
                      size_t matrix_rows,
                      double phase_miss,
                      double cum_miss,
                      double shift,
                      double shift_threshold,
                      double weight_delta,
                      const std::vector<double>& weights);

    void log_learning(uint64_t seq,
                      const char* action,   // upsert_new / upsert_replace / upsert_skip / trim / phase
                      size_t matrix_rows,
                      double phase_miss,
                      double dist_to_nearest,
                      int dominant_arm,
                      const std::vector<double>& weights);

    void flush();

private:
    bool enabled_ = false;
    uint64_t interval_ = 10000;
    uint8_t arm_count_ = 4;
    std::string prefix_;
    std::ofstream progress_file_;
    std::ofstream learning_file_;
    bool progress_header_ = false;
    bool learning_header_ = false;

    void ensure_progress_header();
    void ensure_learning_header();
    static bool ensure_parent_dir(const std::string& filepath);
};

#endif // TRAIN_LOGGER_HPP
