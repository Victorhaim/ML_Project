#ifndef WORKLOAD_PROFILER_HPP
#define WORKLOAD_PROFILER_HPP

#include <vector>
#include <deque>
#include <unordered_map>
#include <cstdint>
#include <string>
#include "TrainLogger.hpp"

enum class ProfilerMode {
    TRAIN,
    TEST
};

struct WorkloadFeatures {
    double write_ratio = 0.0;
    double avg_request_size = 0.0;
    double size_variance = 0.0;
    double singleton_ratio = 0.0;
    double popularity_skewness = 0.0;
    double avg_frequency = 0.0;
    double object_diversity = 0.0;
    double avg_reuse_distance = 0.0;
    double sequentiality_ratio = 0.0;
    double out_cache_hit_rate = 0.0;
    // Time features are expressed relative to the run's own typical
    // inter-arrival gap, so they do not depend on the trace timestamp unit.
    double request_rate = 0.0;          // dt_scale / mean_dt (relative intensity)
    double burstiness_index = 0.0;      // std(dt) / mean(dt)
    double arrival_time_variance = 0.0; // var(dt) / dt_scale^2
    double working_set_byte_delta = 0.0;
    double scan_ratio = 0.0;
};

struct RequestLog {
    uint64_t id;
    uint32_t size;
    bool is_write;
    double timestamp;
    bool out_cache_hit;

    double reuse_dist;
    bool is_sequential;
    double dt;
    bool first_time;
};

// Stored regime: min-max feature box + committed arm weights + measured
// in-process delta against the exactly aligned ThreeLCache shadow.
struct PolicyRecord {
    WorkloadFeatures features;
    WorkloadFeatures feat_min;
    WorkloadFeatures feat_max;
    bool has_box = false;
    std::vector<double> weights;
    double best_miss_rate = 1.0;
    // Positive means MAB beat the in-process ThreeLCache shadow.
    double best_delta = 0.0;
};

class WorkloadProfiler {
private:
    static constexpr double WIN_DELTA_MARGIN = 0.005; // 0.5% absolute BMR gain
    static constexpr double MAX_WEIGHT_RATIO = 100.0;
    // Storage may keep strongly-locked winning arms; injection stays stricter.
    static constexpr double MAX_STORABLE_WEIGHT_RATIO = 1000.0;
    // Dynamic-regime boundaries: feature ε from anchor, byte shock, or max len.
    // Arm thrash must NOT close a regime (arm clock is separate).
    static constexpr size_t   MAX_PHASE_LEN = 50000;
    static constexpr double   SHOCK_SIZE_MULT = 8.0;   // req size vs window avg
    static constexpr size_t   SHOCK_MIN_PHASE = 1000;  // min phase len after a shock
    // Debounce for event-driven closes only; storing a policy has no min length.
    static constexpr size_t   MIN_STORE_LEN = 2000;
    // The EWMA variance of a stable feature decays toward zero, which blows up
    // every normalized distance and drags the regime threshold with it. Floor
    // the per-feature sigma both absolutely and relative to the feature scale.
    static constexpr double   FEATURE_VAR_FLOOR = 1e-4;
    static constexpr double   FEATURE_STD_REL_FLOOR = 0.02;
    // One wild feature must not dominate the aggregate distance.
    static constexpr double   MAX_NORM_DIFF = 10.0;
    static constexpr double   MIN_SHIFT_THRESHOLD = 0.01;
    static constexpr double   MAX_SHIFT_THRESHOLD = 3.0;
    static constexpr double   MIN_MATRIX_EPSILON = 0.005;
    static constexpr double   MAX_MATRIX_EPSILON = 1.0;
    static constexpr uint64_t TEST_MATCH_RETRY = 1000;
    // A regime closes on feature stability, so its stored min/max box only
    // spans the noise of a deliberately steady window: 2-6% wide on average
    // and exactly zero on a third of the rows. Demanding all ten structural
    // features inside one such box is unreachable - measured 0 of 1629
    // held-out regimes, and 42% of training rows fail it against each other.
    // Pad by sigma and accept a majority instead of the full conjunction;
    // the TEST distance ceiling is what keeps replay from drifting.
    static constexpr double   POINT_BOX_PAD = 2.0;
    static constexpr int      MATCH_MIN_STRUCTURAL = 8; // of 10 structural
    // Slow EMA: the gap scale should follow the trace, not a single burst.
    static constexpr double   DT_SCALE_ALPHA = 1e-4;

    ProfilerMode current_mode;
    size_t window_size;
    std::deque<RequestLog> window;

    std::unordered_map<uint64_t, size_t> id_counts;
    std::unordered_map<uint64_t, uint32_t> unique_id_sizes;
    std::unordered_map<uint64_t, uint64_t> last_seen_seq;

    WorkloadFeatures current_features;
    WorkloadFeatures phase_start_features; // regime anchor
    WorkloadFeatures phase_box_min;
    WorkloadFeatures phase_box_max;
    bool phase_box_init = false;
    WorkloadFeatures prev_window_features;

    std::string policy_filename;
    std::string config_filename;

    size_t flush_interval;
    uint64_t request_counter;
    uint8_t current_mab_k = 4;

    double shift_threshold = 0.05;
    double matrix_epsilon  = 0.02;
    double ema_delta_mean;
    double ema_delta_var;
    const double alpha = 0.05;

    // Hard containment is required only on features whose scale is set by the
    // access pattern itself. Op/time/sequence features depend on the trace
    // file format, so they contribute to distance but must not veto a match.
    static constexpr bool MATCH_STRUCTURAL[15] = {
        false, // write_ratio: no op column is mapped, so it is constant
        true,  // avg_request_size
        true,  // size_variance
        true,  // singleton_ratio
        true,  // popularity_skewness
        true,  // avg_frequency
        true,  // object_diversity
        true,  // avg_reuse_distance
        false, // sequentiality_ratio: needs offset-style ids to be meaningful
        true,  // out_cache_hit_rate
        false, // request_rate
        false, // burstiness_index
        false, // arrival_time_variance
        true,  // working_set_byte_delta
        true   // scan_ratio
    };

    const double feature_importance_weights[15] = {
        0.0, // write_ratio: constant until an op column is mapped
        1.5, // avg_request_size
        1.0, // size_variance
        1.5, // singleton_ratio
        3.0, // popularity_skewness
        2.0, // avg_frequency
        1.5, // object_diversity
        3.0, // avg_reuse_distance
        2.0, // sequentiality_ratio
        2.0, // out_cache_hit_rate
        0.5, // request_rate
        0.5, // burstiness_index
        0.5, // arrival_time_variance
        1.5, // working_set_byte_delta
        2.5  // scan_ratio
    };

    double feature_means[15];
    double feature_vars[15];
    // True when means/vars were restored from profiler_config (train scale).
    // TEST freezes this ruler; TRAIN may keep adapting after load.
    bool feature_scale_loaded = false;

    uint64_t phase_requests;
    uint64_t phase_misses;
    uint64_t phase_bytes = 0;
    uint64_t phase_miss_bytes = 0;
    uint64_t phase_shadow_misses = 0;
    uint64_t phase_shadow_miss_bytes = 0;
    // TRAIN upsert score: only requests after Exp3 commit. Investigate
    // traffic must not tax (or inflate) the stored policy's shadow delta.
    bool phase_score_active = false;
    uint64_t score_requests = 0;
    uint64_t score_bytes = 0;
    uint64_t score_miss_bytes = 0;
    uint64_t score_shadow_miss_bytes = 0;
    uint64_t train_total_reqs = 0;
    uint64_t train_total_misses = 0;
    WorkloadFeatures last_stable_features;
    std::vector<double> last_stable_weights;

    uint64_t local_seq;
    uint64_t last_id;
    uint32_t last_size;
    double last_timestamp;
    // Running scale of inter-arrival gaps; divides the raw time features so
    // seconds-based and tick-based traces produce comparable values.
    double dt_scale;
    uint64_t last_match_attempt_seq = 0;
    bool active_policy_match = false;
    bool active_match_refused = false;
    int active_match_band = 6; // unmatched
    double active_match_dist = 1e300;
    int active_policy_row = -1;
    int active_policy_arm = -1;
    int active_policy_type = 10;
    WorkloadFeatures active_inject_features;
    std::vector<double> active_policy_weights;
    double test_match_max_dist = 3.5;

    size_t write_count;
    double size_sum;
    double size_sq_sum;
    size_t singleton_count;
    double freq_sq_sum;
    double total_reuse_dist;
    size_t reuse_count;
    size_t sequential_count;
    size_t out_cache_hits;
    double dt_sum;
    double dt_sq_sum;
    size_t dt_count;
    double working_set_bytes;
    size_t first_time_seen_count;

    std::vector<PolicyRecord> policy_matrix;
    TrainLogger train_log;

    // Compact regime statistics by request-size area. Events are appended and
    // later aggregated into regime_stats.csv (not a per-regime report).
    static constexpr int N_SIZE_AREAS = 8;
    struct RegimeCloseEvent {
        double avg_size = 0.0;
        uint64_t len = 0;
        uint64_t bytes = 0;
        bool long_enough = false;
        bool arm_chosen = false;
        double delta_bmr = 0.0;
        bool stored = false;
        int type_idx = 0; // behavior type for type_stats.csv
    };
    std::vector<RegimeCloseEvent> regime_events;
    std::string regime_events_path;
    std::string regime_stats_path;
    std::string type_stats_path;

    // Behavior types (orthogonal to request-size areas).
    static constexpr int N_REGIME_TYPES = 11; // 10 named + mixed
    static int classify_regime_type(const WorkloadFeatures& f);
    static const char* regime_type_name(int idx);

    static int size_area_index(double avg_size);
    static const char* size_area_name(int idx);
    void note_regime_close(const char* action, double delta_bmr);
    void flush_regime_events();
    void rewrite_regime_stats() const;
    void rewrite_type_stats() const;
    void flush_feature_samples();
    void rewrite_feature_stats() const;

    // Held-out TEST statistics: closest policy-box distance band × size area.
    // Bands: in_box, <=0.02, 0.02-0.05, 0.05-0.15, 0.15-0.50,
    // >0.50, unmatched.
    static constexpr int N_MATCH_BANDS = 7;
    struct TestMatchEvent {
        double avg_size = 0.0;
        uint64_t len = 0;
        uint64_t bytes = 0;
        int match_band = 6;
        bool policy_applied = false;
        bool match_refused = false;
        double distance = 1e300;
        double close_distance = 1e300;
        double delta_bmr = 0.0;
        int type_idx = 10; // behavior type for test type_stats.csv
        int inject_type_idx = 10;
        int policy_type_idx = 10;
        int policy_row = -1;
        int policy_arm = -1;
        std::string close_reason;
        std::vector<double> policy_weights;
        WorkloadFeatures inject_features;
        WorkloadFeatures close_features;
    };
    std::vector<TestMatchEvent> test_match_events;
    std::string test_match_events_path;
    std::string test_match_stats_path;
    std::string test_type_stats_path;
    static int match_band_index(bool in_box, bool usable, double distance);
    static const char* match_band_name(int idx);
    void note_test_match_close(double delta_bmr, uint64_t win_requests,
                               uint64_t win_bytes, const char* close_reason);
    void flush_test_match_events();
    void rewrite_test_match_stats() const;
    void rewrite_test_type_stats() const;
    void set_test_match_stats_paths(const std::string& events_path,
                                    const std::string& stats_path);
    void set_test_type_stats_path(const std::string& stats_path);

    // Feature-mass samples (one vector per closed regime) for p5/p95 coverage.
    static constexpr int N_FEATURES = 15;
    static const char* feature_name(int idx);
    std::vector<std::vector<double>> feature_samples;
    std::string feature_samples_path;
    std::string feature_stats_path;

    void recalculate_features();
    void update_feature_statistics(const WorkloadFeatures& f);
    double feature_std(size_t i) const;
    void clamp_thresholds();
    std::vector<double> to_vector(const WorkloadFeatures& f) const;
    void from_vector(const std::vector<double>& v, WorkloadFeatures& f) const;
    void expand_box(WorkloadFeatures& mn, WorkloadFeatures& mx, const WorkloadFeatures& f) const;
    double box_volume(const WorkloadFeatures& mn, const WorkloadFeatures& mx) const;
    bool box_inside_box(const WorkloadFeatures& inner_min,
                        const WorkloadFeatures& inner_max,
                        const WorkloadFeatures& outer_min,
                        const WorkloadFeatures& outer_max) const;
    bool point_in_box(const WorkloadFeatures& p,
                      const WorkloadFeatures& mn,
                      const WorkloadFeatures& mx) const;
    bool record_contains(const WorkloadFeatures& p, const PolicyRecord& rec) const;
    double calculate_aggregate_distance(const WorkloadFeatures& f1, const WorkloadFeatures& f2);
    void update_online_threshold(double current_delta);
    void save_config();
    void load_config();
    void save_policy_matrix();
    void load_policy_matrix();
    void seed_initial_policies();
    void normalize_weights(std::vector<double>& weights) const;
    bool weights_are_sane(const std::vector<double>& weights) const;
    bool weights_are_storable(const std::vector<double>& weights) const;
    // Removes only coverage-redundant records. There is deliberately no
    // fixed row cap: distinct winning feature regions must remain covered.
    void compact_policy_matrix();
    void open_new_regime();
    const char* close_current_regime(const char* reason, double* out_delta = nullptr);
    void write_features_csv(std::ostream& file, const WorkloadFeatures& f) const;
    double last_match_dist = 1e300;
    // Returns action string: upsert_new / upsert_replace / upsert_skip / ...
    const char* upsert_policy(const WorkloadFeatures& center,
                              const WorkloadFeatures& box_min,
                              const WorkloadFeatures& box_max,
                              bool has_box,
                              const std::vector<double>& weights,
                              double miss_rate,
                              double delta,
                              double* out_dist = nullptr);
    // Prefer the closest containing box; if none contains the point, replay
    // the closest valid row and classify it by normalized feature distance.
    std::vector<double> find_closest_policy(const WorkloadFeatures& current,
                                            bool* out_in_box = nullptr,
                                            int* out_policy_row = nullptr);

public:
    WorkloadProfiler();
    ~WorkloadProfiler();

    void init(const std::string& mode_str,
              const std::string& policy_file = "meta_policy_v3.txt",
              const std::string& config_file = "profiler_config_v3.txt",
              size_t cache_size = 10000,
              uint8_t mab_k = 4,
              bool train_log_enable = true,
              const std::string& train_log_prefix = "mab_train",
              uint64_t train_log_interval = 10000);

    // arm_committed: MAB has finished investigate and locked a policy for this
    // feature regime. Only committed regimes are stored.
    // out_regime_reset: feature ε (or shock/max-len) closed the old regime;
    // MAB must restart investigate on the new one.
    // Returns true when TEST injects a stored policy (contained or nearest).
    bool add_request(uint64_t id, uint32_t size, bool is_write, double timestamp,
                     bool out_cache_hit, bool is_miss, bool shadow_is_miss,
                     const std::vector<double>& current_weights,
                     bool arm_committed,
                     std::vector<double>& out_new_weights,
                     bool* out_regime_reset = nullptr);

    void flush_to_disk();
    void set_regime_stats_paths(const std::string& events_path,
                                const std::string& stats_path);
    void set_feature_stats_paths(const std::string& samples_path,
                                 const std::string& stats_path);
    void set_type_stats_path(const std::string& stats_path);

    WorkloadFeatures get_features() const { return current_features; }
    ProfilerMode get_mode() const { return current_mode; }
    size_t matrix_size() const { return policy_matrix.size(); }
    double get_shift_threshold() const { return shift_threshold; }
    void set_test_match_max_dist(double distance);
};

#endif // WORKLOAD_PROFILER_HPP
