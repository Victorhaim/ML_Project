#ifndef WIN_LEDGER_HPP
#define WIN_LEDGER_HPP

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

// Per-window miss accounting, emitted in an identical format by both the MAB
// run and the frozen-baseline run (mab-off=1) so the two files can be aligned
// row-by-row afterwards to locate the windows where the MAB actually won.
//
// Produces: <prefix>_windows.csv
class WinLedger {
public:
    static const int N_FEATURES = 15;

    WinLedger() = default;
    ~WinLedger();

    void init(bool enable,
              const std::string& prefix,
              uint64_t window,
              uint8_t arm_count,
              const std::string& run_tag);

    bool enabled() const { return enabled_; }
    uint64_t window() const { return window_; }

    // features      = end-of-window point (kept for backward compatibility)
    // feat_min/max  = observed range across the window, so an offline confirm
    //                 can build a real min-max regime box instead of a point.
    void log_window(uint64_t win_idx,
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
                    uint64_t shadow_bytes_miss);

    void flush();

private:
    bool enabled_ = false;
    uint64_t window_ = 100000;
    uint8_t arm_count_ = 4;
    std::string prefix_;
    std::string run_tag_;
    std::ofstream file_;
    bool header_written_ = false;

    void ensure_header();
    static bool ensure_parent_dir(const std::string& filepath);
};

#endif // WIN_LEDGER_HPP
