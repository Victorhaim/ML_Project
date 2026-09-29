#ifndef WORKLOAD_PROFILER_HPP
#define WORKLOAD_PROFILER_HPP

#include <vector>
#include <deque>
#include <array>
#include <unordered_map>
#include <cstdint>
#include <string>
#include <cmath>
#include <algorithm>
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

// KNN v5 row: one feature region, one persistent MODE-arm queue, and the
// single best whole-regime run observed for that region.
struct PolicyRecord {
    WorkloadFeatures features;
    WorkloadFeatures feat_min;
    WorkloadFeatures feat_max;
    bool has_box = false;
    int winner_arm = -1;
    double best_miss_rate = 1.0;
    double best_delta = -1e300;
    std::vector<uint8_t> arm_queue;
    uint8_t queue_next = 0;
    // Bit i is set only after arm i completed a TRAIN regime. Queue position
    // alone cannot express coverage after the queue wraps.
    uint16_t tried_arms_mask = 0;
};

class WorkloadProfiler {
private:
    static constexpr double MAX_WEIGHT_RATIO = 100.0;
    // Storage may keep strongly-locked winning arms; injection stays stricter.
    static constexpr double MAX_STORABLE_WEIGHT_RATIO = 1000.0;
    // Dynamic-regime boundaries: feature ε from the anchor, or a byte shock.
    // A regime lasts exactly as long as the behaviour that defines it: there
    // is deliberately no length cap and no minimum length. Arm thrash must NOT
    // close a regime (the arm clock is separate), and an underperforming arm
    // is re-decided on the arm clock rather than by cutting the regime.
    static constexpr double   SHOCK_SIZE_MULT = 8.0;   // req size vs window avg

    // ---- Regime decision clock -------------------------------------------
    // Features are rolling-window averages over ~16k requests, so they cannot
    // meaningfully change between two consecutive requests. Evaluating them
    // per request cost a full feature pass plus a distance computation on
    // every single request and let one noisy sample cut a regime, which is
    // how the median regime length collapsed to one request. Decisions now
    // run on their own clock; the counters behind them stay per-request.
    static constexpr uint64_t REGIME_TICK = 512;      // requests per evaluation
    static constexpr uint32_t DRIFT_CONFIRM = 2;      // ticks above threshold
    static constexpr uint64_t MIN_REGIME_LEN = 2048;  // no close below this
    static constexpr uint64_t MAX_REGIME_LEN = 131072; // forced close
    // A tick counts as drift evidence when this share of it was byte shocks.
    static constexpr double   SHOCK_TICK_SHARE = 0.10;

    // ---- Feature scale ----------------------------------------------------
    // The old ruler was an EWMA of squared per-request deltas. On a rolling
    // window that moves by 1/window per request those deltas are ~0, so the
    // variance decayed onto FEATURE_VAR_FLOOR and the feature was declared
    // dead even when it ranged over the whole trace. Scale now comes from the
    // spread of a reservoir of tick samples, which is immune to how closely
    // consecutive samples are correlated.
    static constexpr size_t   SCALE_RESERVOIR_CAP = 4096;
    static constexpr size_t   SCALE_MIN_SAMPLES = 64;  // before trusting spread
    static constexpr uint64_t SCALE_REFRESH_TICKS = 256;
    // Robust sigma is (p84 - p16) / 2; below this share of |mean| a feature
    // is treated as having no usable spread.
    static constexpr double   FEATURE_SPREAD_REL_EPS = 1e-3;
    // The EWMA variance of a stable feature decays toward zero, which blows up
    // every normalized distance and drags the regime threshold with it. Floor
    // the per-feature sigma both absolutely and relative to the feature scale.
    static constexpr double   FEATURE_VAR_FLOOR = 1e-4;
    static constexpr double   FEATURE_STD_REL_FLOOR = 0.02;
    // One wild feature must not dominate the aggregate distance.
    static constexpr double   MAX_NORM_DIFF = 10.0;
    static constexpr double   MIN_SHIFT_THRESHOLD = 0.01;
    static constexpr double   MAX_SHIFT_THRESHOLD = 3.0;
    // How many σ of tick-to-tick distance close a regime. 3.0 only fires on
    // rare jumps; 2.0 splits slow drift into more scored boxes for KNN.
    static constexpr double   DEFAULT_SHIFT_SIGMA = 2.0;
    static constexpr double   MIN_SHIFT_SIGMA = 1.0;
    static constexpr double   MAX_SHIFT_SIGMA = 4.0;
    static constexpr double   MIN_MATRIX_EPSILON = 0.005;
    static constexpr double   MAX_MATRIX_EPSILON = 1.0;
    static constexpr uint64_t TEST_MATCH_RETRY = 1000;
    // A row is the min and max each structural feature took during the first
    // MIN_REGIME_LEN requests. Outside that range is a different area: no pad.
    // TEST uses a soft distance vote, so distant rows fade rather than being
    // accepted/rejected at a hard threshold.
    // In units of the feature's robust sigma. While feature_std was pinned at
    // the variance floor a pad of 2.0 sigma was a hair in feature units; with
    // a real ruler the same multiplier exceeds the p5-p95 span of four of the
    // seven matchable features, so every regime lands inside every box.
    static constexpr double   DEFAULT_BOX_PAD = 0.1;
    // Hard containment (open-match and skip_covered) uses only features whose
    // scale is set by the access pattern. Time/op/sequence coordinates may
    // still enter KNN distance when the screen mask enables them, but they
    // must not veto "we have seen this regime."
    static constexpr bool MATCH_STRUCTURAL[15] = {
        false, // write_ratio
        true,  // avg_request_size
        true,  // size_variance
        true,  // singleton_ratio
        true,  // popularity_skewness
        true,  // avg_frequency
        true,  // object_diversity
        true,  // avg_reuse_distance
        false, // sequentiality_ratio
        true,  // out_cache_hit_rate
        false, // request_rate
        false, // burstiness_index
        false, // arrival_time_variance
        true,  // working_set_byte_delta
        true   // scan_ratio
    };
    // Slow EMA: the gap scale should follow the trace, not a single burst.
    static constexpr double   DT_SCALE_ALPHA = 1e-4;

    ProfilerMode current_mode;
    size_t window_size;
    std::deque<RequestLog> window;
    bool profile_ready_seen = false;

    std::unordered_map<uint64_t, size_t> id_counts;
    std::unordered_map<uint64_t, uint32_t> unique_id_sizes;
    std::unordered_map<uint64_t, uint64_t> last_seen_seq;

    WorkloadFeatures current_features;
    WorkloadFeatures phase_start_features; // regime anchor
    WorkloadFeatures phase_box_min;
    WorkloadFeatures phase_box_max;
    bool phase_box_init = false;
    // Min/max stop growing once the regime has its minimum length. Later
    // requests still score the arm; they do not widen the area.
    bool phase_box_frozen = false;
    bool phase_profile_ready = false;
    WorkloadFeatures prev_window_features;

    std::string policy_filename;
    std::string config_filename;

    size_t flush_interval;
    uint64_t request_counter;
    uint8_t current_mab_k = 6;

    double shift_threshold = 0.05;
    double matrix_epsilon  = 0.02;
    double shift_sigma = DEFAULT_SHIFT_SIGMA;
    double ema_delta_mean;
    double ema_delta_var;
    const double alpha = 0.05;

    // Default keeps historical behavior: all 15 coordinates participate.
    // The feature-screen shell passes a 15-bit mask to override this per run.
    std::array<bool, 15> feature_active{{
        true, true, true, true, true, true, true, true,
        true, true, true, true, true, true, true
    }};

    const double feature_importance_weights[15] = {
        1.0, // write_ratio
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
    // Robust 1-sigma per feature, derived from scale_reservoir spread.
    // feature_vars is kept only so the v2 config format still round-trips.
    double feature_scale[15];
    // Scale as restored from config. TRAIN widens the ruler to cover every
    // trace seen so far instead of overwriting it with the current trace's
    // spread, so the ruler TEST inherits spans the whole training corpus.
    double loaded_scale[15];
    // Set once the reservoir has produced a real spread. Until then the old
    // EWMA ruler bootstraps means/vars so warmup distances stay finite.
    bool scale_ready = false;
    // True when means/scale were restored from profiler_config (train scale).
    // TEST freezes this ruler; TRAIN may keep adapting after load.
    bool feature_scale_loaded = false;

    // Indices of the features this run's mask enabled. Everything that walks
    // features - recalculation, distance, containment, the stored box - walks
    // this list, so an unmasked feature costs nothing anywhere.
    std::vector<size_t> active_idx;
    // Set when the only consumers of an expensive per-request counter are
    // masked off, so the counter itself can be skipped.
    bool need_reuse_distance = true;   // last_seen_seq map
    bool need_working_set = true;      // unique_id_sizes map
    bool need_time_features = true;    // dt accumulators
    void rebuild_active_index();
    std::unordered_map<std::string, uint64_t> action_counts;

    // Containment pad in sigma. This was tuned when feature_std was pinned
    // near the variance floor; with a real spread the same multiplier makes
    // every box far wider, so it is now tunable.
    double box_pad = DEFAULT_BOX_PAD;
    // The run's mask as 15 chars, written into the policy header so a table
    // built for a different feature set is rejected instead of misparsed.
    std::string feature_mask_str = "111111111111111";
    void write_active_csv(std::ostream& file, const WorkloadFeatures& f) const;

    // Tick-sampled feature vectors backing the robust scale.
    std::vector<std::vector<double>> scale_reservoir;
    uint64_t scale_seen = 0;
    uint64_t ticks_since_scale = 0;
    void scale_reservoir_add(const std::vector<double>& v);
    void refresh_feature_scale();

    // Regime decision clock.
    uint64_t tick_counter = 0;
    uint32_t drift_ticks = 0;
    uint64_t phase_ticks = 0;
    uint64_t phase_shock_count = 0;
    // Rolling average request size, maintained regardless of the mask so byte
    // shock detection survives avg_request_size being masked off.
    double shock_ref_size = 0.0;

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
    std::string last_regime_close_reason;
    bool active_policy_match = false;
    bool active_match_refused = false;
    int active_match_band = 6; // unmatched
    double active_match_dist = 1e300;
    // Distances to the five nearest stored rows, nearest first. -1 means
    // that neighbour did not exist. TEST weights use only these distances.
    std::array<double, 5> active_nn_dist{{-1.0, -1.0, -1.0, -1.0, -1.0}};
    int active_policy_row = -1;
    int active_policy_arm = -1;
    int active_policy_type = 10;
    WorkloadFeatures active_inject_features;
    std::vector<double> active_policy_weights;

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
    // Street map: bins on size / reuse / skew. Memory is the DB; the policy
    // file is only a per-trace checkpoint.
    std::unordered_map<uint64_t, std::vector<int>> policy_grid;
    std::vector<int> policy_wide;
    uint64_t policy_tombstones = 0;
    void rebuild_policy_index();
    void index_insert_row(int row);
    int index_bin(size_t feat, double x) const;
    static uint64_t index_pack(int a, int b, int c);
    // Lookup answers "which row covers / is nearest to this point" and runs
    // once per regime open. Cover answers "does a row already subsume this
    // box", runs only inside upsert_policy, and must see every row or the
    // table gains duplicates. Only Cover is forced linear.
    enum class IndexUse { Lookup, Cover };
    bool index_linear_faster(IndexUse use) const;
    bool index_box_too_wide(const WorkloadFeatures& mn,
                            const WorkloadFeatures& mx) const;
    std::vector<int> index_candidates_point(const WorkloadFeatures& p) const;
    std::vector<int> index_candidates_box(const WorkloadFeatures& mn,
                                          const WorkloadFeatures& mx) const;
    TrainLogger train_log;
    int active_train_policy_row = -1;
    int active_train_arm = 3;
    // Rotation used when no stored row covers a regime. Rows are created only
    // by a scored close, so an unmatched open must still spread arms evenly.
    uint8_t train_arm_rotation = 0;

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
    static const char* train_arm_name(int arm);

    static int size_area_index(double avg_size);
    static const char* size_area_name(int idx);
    void note_regime_close(const char* action, double delta_bmr);
    // The small *_stats.csv are formatted from live counters, never by
    // re-reading event dumps, so they exist even with the dumps disabled.
    static constexpr uint64_t STATS_REWRITE_EVERY = 32; // closed regimes
    static constexpr uint64_t FEATURE_STATS_EVERY = 256;
    static constexpr size_t   STATS_SAMPLE_CAP = 8192;    // per reservoir
    uint64_t stat_closes = 0;
    uint64_t stat_last_rewrite = 0;
    uint64_t feature_stat_last_rewrite = 0;
    uint64_t test_stat_closes = 0;
    uint64_t test_stat_last_rewrite = 0;
    uint64_t stats_rng = 0x9E3779B97F4A7C15ULL;

    // Exact until the cap, uniformly sampled after it, so medians and p5/p95
    // stay representative without retaining every regime.
    struct SampleSet {
        std::vector<double> v;
        uint64_t seen = 0;
    };
    void sample_add(SampleSet& s, double x);

    struct TrainAgg {
        uint64_t regimes = 0, bytes = 0, long_enough = 0, arm_chosen = 0;
        uint64_t beat = 0, lost = 0, stored = 0, delta_n = 0;
        uint64_t at_min_len = 0, drift_close = 0, max_len_close = 0;
        double   delta_sum = 0.0;
        SampleSet deltas;
        SampleSet lens;
    };
    TrainAgg train_band[N_SIZE_AREAS];
    TrainAgg train_all;
    // TRAIN type_stats: one cell per arm × feature type. The last type slot
    // is ALL types for that arm. Only a regime that actually played an arm
    // is counted.
    static constexpr int N_TRAIN_ARMS = 6;
    struct ArmTypeAgg {
        uint64_t regimes = 0, bytes = 0, won = 0, lost = 0, draw = 0;
        double win_delta_sum = 0.0, loss_delta_sum = 0.0, byte_delta_sum = 0.0;
    };
    ArmTypeAgg arm_type[N_TRAIN_ARMS][N_REGIME_TYPES + 1];

    std::vector<std::vector<double>> feature_reservoir;
    uint64_t feature_reservoir_seen = 0;

    void flush_regime_events(bool rebuild_stats = false);
    void rewrite_regime_stats() const;
    void rewrite_type_stats() const;
    void flush_feature_samples();
    void rewrite_feature_stats() const;

    // Held-out TEST statistics: nearest-neighbour sigma distance × size area.
    // Bands are distance only: <=0.5, 0.5-1, 1-1.5, 1.5-2.5, 2.5-4, >4,
    // unmatched. Containment is not a band. The five raw distances are in
    // nn_distances.csv.
    static constexpr int N_MATCH_BANDS = 7;
    struct TestAgg {
        uint64_t regimes = 0, bytes = 0, applied = 0, applied_bytes = 0;
        uint64_t refused = 0, refused_bytes = 0, won = 0, lost = 0, arm_chosen = 0;
        double delta_sum = 0.0, byte_delta_sum = 0.0;
        SampleSet deltas;
    };
    TestAgg test_band[N_MATCH_BANDS][N_SIZE_AREAS + 1];
    TestAgg test_grand;
    TestAgg test_type[N_REGIME_TYPES + 1][N_SIZE_AREAS + 1];
    struct TestMatchEvent {
        double avg_size = 0.0;
        uint64_t len = 0;
        uint64_t bytes = 0;
        int match_band = 6;
        std::array<double, 5> nn_dist{{-1.0, -1.0, -1.0, -1.0, -1.0}};
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
    std::string nn_distances_path;
    std::string test_type_stats_path;
    static int match_band_index(bool usable, double distance);
    static const char* match_band_name(int idx);
    void note_test_match_close(double delta_bmr, uint64_t win_requests,
                               uint64_t win_bytes, const char* close_reason);
    void flush_test_match_events(bool rebuild_stats = false);
    void rewrite_test_match_stats() const;
    void rewrite_test_type_stats() const;
    void set_test_match_stats_paths(const std::string& events_path,
                                    const std::string& stats_path,
                                    const std::string& nn_path);
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
    // A masked-off feature, or one whose observed spread never rose above the
    // variance floor. A constant coordinate is inside every box and adds zero
    // to the distance numerator, so letting it participate manufactures
    // agreement that the workload never showed.
    bool feature_usable(size_t i) const;
    // Usable AND structural: the backup's match/merge axes, intersected with
    // the experiment mask.
    bool feature_matchable(size_t i) const;
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
                              int arm,
                              double miss_rate,
                              double delta,
                              double* out_dist = nullptr);
    void initialize_arm_queue(PolicyRecord& record);
    // Five-neighbour interpolation over each row's saved winner. A row
    // contributes in proportion to inverse squared sigma distance only.
    // Shadow gain chooses which arm that row stored; it does not vote.
    // The cache converts the resulting distribution into one blended queue.
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
              uint8_t mab_k = 6,
              bool train_log_enable = true,
              const std::string& train_log_prefix = "mab_train",
              uint64_t train_log_interval = 10000,
              const std::string& feature_mask = "111111111111111",
              bool persist_event_logs = true);

    const std::unordered_map<std::string, uint64_t>& get_action_counts() const {
        return action_counts;
    }

    void set_box_pad(double pad) {
        if (!std::isfinite(pad) || pad < 0.0) pad = DEFAULT_BOX_PAD;
        box_pad = pad;
    }

    void set_shift_sigma(double sigma) {
        if (!std::isfinite(sigma)) sigma = DEFAULT_SHIFT_SIGMA;
        shift_sigma = std::min(MAX_SHIFT_SIGMA, std::max(MIN_SHIFT_SIGMA, sigma));
    }

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
    int current_regime_type_idx() const {
        return classify_regime_type(current_features);
    }
    std::vector<double> get_feature_vector() const {
        return to_vector(current_features);
    }
    bool is_profile_ready() const { return window.size() >= window_size; }
    const std::string& get_last_regime_close_reason() const {
        return last_regime_close_reason;
    }
    ProfilerMode get_mode() const { return current_mode; }
    size_t matrix_size() const { return policy_matrix.size(); }
    double get_shift_threshold() const { return shift_threshold; }
    uint8_t select_regime_arm(bool* out_matched = nullptr);
    const std::vector<double>& get_active_policy_weights() const {
        return active_policy_weights;
    }
    // MODEL TEST: record the first-tree start so match/type stats still
    // have an arm and mix without KNN inject.
    void set_window_start_audit(int start_arm,
                                const std::vector<double>& start_arc);
};

#endif // WORKLOAD_PROFILER_HPP
