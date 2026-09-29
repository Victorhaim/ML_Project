#include "WorkloadProfiler.hpp"
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace {

bool ensure_parent_dir(const std::string& filepath) {
    if (filepath.empty()) return false;
    std::filesystem::path parent = std::filesystem::path(filepath).parent_path();
    if (parent.empty()) return true;
    std::error_code ec;
    if (std::filesystem::exists(parent, ec)) return true;
    return std::filesystem::create_directories(parent, ec);
}

} // namespace

WorkloadProfiler::WorkloadProfiler() : 
    current_mode(ProfilerMode::TRAIN), window_size(20000),
    flush_interval(0), request_counter(0), current_mab_k(6),
    shift_threshold(0.05), matrix_epsilon(0.02),
    shift_sigma(DEFAULT_SHIFT_SIGMA),
    ema_delta_mean(0.02), ema_delta_var(0.0004),
    phase_requests(0), phase_misses(0), local_seq(0), last_id(0), last_size(0),
    last_timestamp(0.0), dt_scale(0.0),
    write_count(0), size_sum(0.0), size_sq_sum(0.0), singleton_count(0),
    freq_sq_sum(0.0), total_reuse_dist(0.0), reuse_count(0), sequential_count(0),
    out_cache_hits(0), dt_sum(0.0), dt_sq_sum(0.0), dt_count(0),
    working_set_bytes(0.0), first_time_seen_count(0) {
    
    for (int i = 0; i < 15; ++i) {
        feature_means[i] = 0.0;
        feature_vars[i]  = 1.0;
        feature_scale[i] = 0.0;
        loaded_scale[i]  = 0.0;
    }
    rebuild_active_index();
    current_features = WorkloadFeatures();
    phase_start_features = WorkloadFeatures();
    phase_box_min = WorkloadFeatures();
    phase_box_max = WorkloadFeatures();
    phase_box_init = false;
    prev_window_features = WorkloadFeatures();
}

WorkloadProfiler::~WorkloadProfiler() {
    if (current_mode == ProfilerMode::TEST) {
        double final_delta = 0.0;
        close_current_regime("run_end", &final_delta);
        flush_test_match_events(true);
        return;
    }
    double final_delta = 0.0;
    const char* final_action =
        close_current_regime("run_end", &final_delta);
    if (train_log.enabled()) {
        train_log.log_learning(local_seq, final_action,
                               policy_matrix.size(), final_delta,
                               0.0, -1, last_stable_weights);
        train_log.flush();
    }
    flush_regime_events(true);
    rewrite_feature_stats();
        save_policy_matrix();
        save_config();
    }

void WorkloadProfiler::normalize_weights(std::vector<double>& weights) const {
    if (weights.empty()) return;
    double max_w = 0.0;
    for (double w : weights) if (w > max_w) max_w = w;
    if (max_w <= 0.0) {
        for (double& w : weights) w = 1.0;
        max_w = 1.0;
    }
    const double min_allowed = max_w * 1e-3;
    for (double& w : weights) {
        if (w < min_allowed) w = min_allowed;
    }
    double sum = 0.0;
    for (double w : weights) sum += w;
    if (sum <= 0.0) return;
    const double target = static_cast<double>(weights.size());
    for (double& w : weights) w = w / sum * target;
}

bool WorkloadProfiler::weights_are_sane(const std::vector<double>& weights) const {
    if (weights.empty()) return false;
    double mx = weights[0], mn = weights[0];
    for (double w : weights) {
        if (!(w > 0.0) || !std::isfinite(w)) return false;
        if (w > mx) mx = w;
        if (w < mn) mn = w;
    }
    if (mn <= 0.0) return false;
    return (mx / mn) <= MAX_WEIGHT_RATIO;
}

bool WorkloadProfiler::weights_are_storable(const std::vector<double>& weights) const {
    if (weights.empty()) return false;
    double mx = weights[0], mn = weights[0];
    for (double w : weights) {
        if (!(w > 0.0) || !std::isfinite(w)) return false;
        if (w > mx) mx = w;
        if (w < mn) mn = w;
    }
    if (mn <= 0.0) return false;
    return (mx / mn) <= MAX_STORABLE_WEIGHT_RATIO;
}

// Drop coverage-dominated rows that were only tombstoned. Queues on live
// rows stay put; this is not a merge.
void WorkloadProfiler::compact_policy_matrix() {
    if (policy_tombstones == 0) return;
    std::vector<PolicyRecord> kept;
    kept.reserve(policy_matrix.size());
    for (const auto& rec : policy_matrix) {
        if (rec.winner_arm < 0) continue;
        kept.push_back(rec);
    }
    policy_matrix.swap(kept);
    policy_tombstones = 0;
    if (active_train_policy_row >= static_cast<int>(policy_matrix.size())) {
        active_train_policy_row = policy_matrix.empty()
            ? -1 : static_cast<int>(policy_matrix.size() - 1);
    }
    rebuild_policy_index();
}

std::vector<double> WorkloadProfiler::to_vector(const WorkloadFeatures& f) const {
    return {
        f.write_ratio, f.avg_request_size, f.size_variance,
        f.singleton_ratio, f.popularity_skewness, f.avg_frequency,
        f.object_diversity, f.avg_reuse_distance, f.sequentiality_ratio,
        f.out_cache_hit_rate, f.request_rate, f.burstiness_index,
        f.arrival_time_variance, f.working_set_byte_delta, f.scan_ratio
    };
}

void WorkloadProfiler::from_vector(const std::vector<double>& v, WorkloadFeatures& f) const {
    if (v.size() < 15) return;
    f.write_ratio = v[0];
    f.avg_request_size = v[1];
    f.size_variance = v[2];
    f.singleton_ratio = v[3];
    f.popularity_skewness = v[4];
    f.avg_frequency = v[5];
    f.object_diversity = v[6];
    f.avg_reuse_distance = v[7];
    f.sequentiality_ratio = v[8];
    f.out_cache_hit_rate = v[9];
    f.request_rate = v[10];
    f.burstiness_index = v[11];
    f.arrival_time_variance = v[12];
    f.working_set_byte_delta = v[13];
    f.scan_ratio = v[14];
}

void WorkloadProfiler::expand_box(WorkloadFeatures& mn, WorkloadFeatures& mx,
                                  const WorkloadFeatures& f) const {
    auto a = to_vector(mn);
    auto b = to_vector(mx);
    auto c = to_vector(f);
    for (size_t i = 0; i < 15; ++i) {
        if (c[i] < a[i]) a[i] = c[i];
        if (c[i] > b[i]) b[i] = c[i];
    }
    from_vector(a, mn);
    from_vector(b, mx);
}

double WorkloadProfiler::box_volume(const WorkloadFeatures& mn,
                                    const WorkloadFeatures& mx) const {
    auto a = to_vector(mn);
    auto b = to_vector(mx);
    double vol = 1.0;
    int active_count = 0;
    for (size_t i = 0; i < 15; ++i) {
        if (feature_usable(i)) active_count++;
    }
    for (size_t i = 0; i < 15; ++i) {
        if (!feature_usable(i)) continue;
        double std_dev = feature_std(i);
        double span = std::max(1e-6, (b[i] - a[i]) / std_dev);
        // Geometric mean of spans keeps volume numerically sane.
        vol *= std::pow(span, 1.0 / std::max(1, active_count));
    }
    return vol;
}

bool WorkloadProfiler::box_inside_box(const WorkloadFeatures& inner_min,
                                      const WorkloadFeatures& inner_max,
                                      const WorkloadFeatures& outer_min,
                                      const WorkloadFeatures& outer_max) const {
    auto imn = to_vector(inner_min);
    auto imx = to_vector(inner_max);
    auto omn = to_vector(outer_min);
    auto omx = to_vector(outer_max);
    for (size_t i : active_idx) {
        if (!feature_matchable(i)) continue;
        if (imn[i] < omn[i] || imx[i] > omx[i]) return false;
    }
    return true;
}

bool WorkloadProfiler::point_in_box(const WorkloadFeatures& p,
                                    const WorkloadFeatures& mn,
                                    const WorkloadFeatures& mx) const {
    auto pv = to_vector(p);
    auto lo = to_vector(mn);
    auto hi = to_vector(mx);
    int match_count = 0;
    for (size_t i : active_idx) {
        if (!feature_matchable(i)) continue;
        match_count++;
        if (pv[i] < lo[i] || pv[i] > hi[i]) return false;
    }
    return match_count > 0;
}

bool WorkloadProfiler::record_contains(const WorkloadFeatures& p,
                                       const PolicyRecord& rec) const {
    if (rec.has_box) {
        return point_in_box(p, rec.feat_min, rec.feat_max);
    }
    // Backward-compatible degenerate point row.
    double dist = 0.0;
    {
        // Local copy of distance without mutating last_match_dist.
        auto v1 = to_vector(p);
        auto v2 = to_vector(rec.features);
        double weighted_sum = 0.0;
        double total_weight = 0.0;
        for (size_t i : active_idx) {
            if (!feature_usable(i)) continue;
            double norm_diff = std::min(MAX_NORM_DIFF,
                                        std::abs(v1[i] - v2[i]) / feature_std(i));
            weighted_sum += feature_importance_weights[i] * norm_diff;
            total_weight += feature_importance_weights[i];
        }
        dist = total_weight > 0.0 ? weighted_sum / total_weight : 1e300;
    }
    return dist <= std::max(matrix_epsilon, 1e-6);
}

void WorkloadProfiler::open_new_regime() {
    phase_start_features = current_features;
    phase_box_min = current_features;
    phase_box_max = current_features;
    phase_box_init = true;
    phase_box_frozen = false;
    phase_profile_ready = profile_ready_seen;
    drift_ticks = 0;
    phase_ticks = 0;
    phase_shock_count = 0;
    phase_requests = 0;
    phase_misses = 0;
    phase_bytes = 0;
    phase_miss_bytes = 0;
    phase_shadow_misses = 0;
    phase_shadow_miss_bytes = 0;
    phase_score_active = false;
    score_requests = 0;
    score_bytes = 0;
    score_miss_bytes = 0;
    score_shadow_miss_bytes = 0;
    last_stable_weights.clear();
    active_policy_match = false;
    active_match_refused = false;
    active_match_band = N_MATCH_BANDS - 1;
    active_match_dist = 1e300;
    active_nn_dist.fill(-1.0);
    active_policy_row = -1;
    active_policy_arm = -1;
    active_policy_type = N_REGIME_TYPES - 1;
    active_policy_weights.clear();
    active_train_policy_row = -1;
    active_train_arm = 3;
}

void WorkloadProfiler::initialize_arm_queue(PolicyRecord& record) {
    const int n = std::max(1, static_cast<int>(current_mab_k));
    record.arm_queue.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        record.arm_queue[static_cast<size_t>(i)] = static_cast<uint8_t>(i);
    }
    // Stable FNV-style hash of the feature center: repeatable across runs,
    // unlike process-global rand(), but different regions get different order.
    uint64_t state = 1469598103934665603ULL;
    for (double value : to_vector(record.features)) {
        uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "double size");
        std::memcpy(&bits, &value, sizeof(bits));
        state ^= bits;
        state *= 1099511628211ULL;
    }
    for (int i = n - 1; i > 0; --i) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        const int j = static_cast<int>((state * 2685821657736338717ULL) %
                                       static_cast<uint64_t>(i + 1));
        std::swap(record.arm_queue[static_cast<size_t>(i)],
                  record.arm_queue[static_cast<size_t>(j)]);
    }
    record.queue_next = 0;
}

const char* WorkloadProfiler::upsert_policy(const WorkloadFeatures& center,
                                            const WorkloadFeatures& box_min,
                                            const WorkloadFeatures& box_max,
                                            bool has_box,
                                            int arm,
                                            double miss_rate,
                                            double delta,
                                            double* out_dist) {
    if (arm < 0 || arm >= static_cast<int>(current_mab_k)) {
        if (out_dist) *out_dist = -1.0;
        return "upsert_reject";
    }

    double min_dist = 1e300;
    auto consider_near = [&](int i) {
        if (i < 0 || i >= static_cast<int>(policy_matrix.size())) return;
        if (policy_matrix[static_cast<size_t>(i)].winner_arm < 0) return;
        const double dist = calculate_aggregate_distance(
            center, policy_matrix[static_cast<size_t>(i)].features);
        if (dist < min_dist) min_dist = dist;
    };
    if (index_linear_faster(IndexUse::Lookup)) {
        for (size_t i = 0; i < policy_matrix.size(); ++i) consider_near(static_cast<int>(i));
    } else {
        for (int i : index_candidates_point(center)) consider_near(i);
    }
    if (out_dist) *out_dist = (policy_matrix.empty() ? 1e300 : min_dist);

    // Coverage is the row that handed out the arm at open. Re-checking
    // containment at close would treat a drifted window as a new area and
    // store the whole min/max span as a wide box that then swallows later
    // regimes. A new row is created only when open found no containing row.
    int row_idx = active_train_policy_row;
    if (row_idx < 0 || row_idx >= static_cast<int>(policy_matrix.size()) ||
        policy_matrix[static_cast<size_t>(row_idx)].winner_arm < 0) {
        double best = 1e300;
        row_idx = -1;
        auto consider_contain = [&](int i) {
            if (i < 0 || i >= static_cast<int>(policy_matrix.size())) return;
            if (!record_contains(center, policy_matrix[static_cast<size_t>(i)])) return;
            const double dist = calculate_aggregate_distance(
                center, policy_matrix[static_cast<size_t>(i)].features);
            if (dist < best) {
                best = dist;
                row_idx = i;
            }
        };
        if (index_linear_faster(IndexUse::Lookup)) {
            for (size_t i = 0; i < policy_matrix.size(); ++i) consider_contain(static_cast<int>(i));
        } else {
            for (int i : index_candidates_point(center)) consider_contain(i);
        }
    }

    // Row existence is geographic, never a reward gate. If this regime
    // opened from an existing row, always record that the arm was tried.
    // Replace the row's policy only when this arm's measured score is
    // better; a loss is still useful arm evidence.
    //
    // The box is deliberately not stretched on a revisit: growing one row
    // lets it swallow the feature space and prevents genuinely new areas from
    // receiving their own rows.
    if (row_idx >= 0) {
        PolicyRecord& src = policy_matrix[static_cast<size_t>(row_idx)];
        src.tried_arms_mask |= static_cast<uint16_t>(1u << arm);
        if (src.winner_arm < 0 || delta > src.best_delta) {
            src.winner_arm = arm;
            src.best_miss_rate = miss_rate;
            src.best_delta = delta;
            return "upsert_update_better";
        }
        return "upsert_update_keep";
    }

    // No row contains this area: create it regardless of win or loss. Its
    // delta is data attached to the area, not permission for the area to
    // exist.
    PolicyRecord rec;
    rec.features = center;
    rec.feat_min = box_min;
    rec.feat_max = box_max;
    rec.has_box = has_box;
    initialize_arm_queue(rec);
    rec.tried_arms_mask = static_cast<uint16_t>(1u << arm);
    // This arm was already consumed by the just-finished occurrence.
    const int n = static_cast<int>(rec.arm_queue.size());
    for (int i = 0; i < n; ++i) {
        if (rec.arm_queue[static_cast<size_t>(i)] ==
            static_cast<uint8_t>(arm)) {
            rec.queue_next = static_cast<uint8_t>((i + 1) % n);
            break;
        }
    }
    rec.winner_arm = arm;
    rec.best_miss_rate = miss_rate;
    rec.best_delta = delta;
    policy_matrix.push_back(rec);
    active_train_policy_row = static_cast<int>(policy_matrix.size() - 1);
    index_insert_row(active_train_policy_row);
    return "upsert_new";
}

const char* WorkloadProfiler::close_current_regime(const char* reason,
                                                   double* out_delta) {
    if (out_delta) *out_delta = 0.0;
    if (phase_requests == 0) {
        return "skip_empty";
    }

    const double mab_bmr = (score_bytes > 0)
        ? static_cast<double>(score_miss_bytes) / score_bytes : 0.0;
    const double shadow_bmr = (score_bytes > 0)
        ? static_cast<double>(score_shadow_miss_bytes) / score_bytes : 0.0;
    const double delta = shadow_bmr - mab_bmr;
    if (out_delta) *out_delta = delta;

    // Held-out TEST measures the replayed policy but never updates the table.
    if (current_mode == ProfilerMode::TEST) {
        // Score only the window where the injected policy actually ran. An
        // unmatched regime never injected, so its whole span is the window.
        const bool scored = phase_score_active && score_bytes > 0;
        const uint64_t win_reqs = scored ? score_requests : phase_requests;
        const uint64_t win_bytes = scored ? score_bytes : phase_bytes;
        const uint64_t win_miss = scored ? score_miss_bytes : phase_miss_bytes;
        const uint64_t win_shadow_miss =
            scored ? score_shadow_miss_bytes : phase_shadow_miss_bytes;

        const double test_mab = (win_bytes > 0)
            ? static_cast<double>(win_miss) / win_bytes : 0.0;
        const double test_shadow = (win_bytes > 0)
            ? static_cast<double>(win_shadow_miss) / win_bytes : 0.0;
        const double test_delta = test_shadow - test_mab;
        if (out_delta) *out_delta = test_delta;
        note_test_match_close(test_delta, win_reqs, win_bytes, reason);
        return active_policy_match ? "test_nn" : "test_unmatched";
    }

    const char* action = "skip_no_score_window";
    // Every scored geographic area is learned. New areas create rows whether
    // they win or lose; covered areas always record the tried arm and replace
    // their winner only when the score improves.
    if (!phase_profile_ready) {
        action = "skip_profile_warmup";
    } else if (!phase_score_active || score_bytes == 0) {
        action = "skip_no_score_window";
    } else if (active_train_arm < 0) {
        action = "skip_uncommitted";
    } else {
        double dist = 0.0;
        action = upsert_policy(last_stable_features,
                               phase_box_min, phase_box_max, true,
                               active_train_arm, mab_bmr, delta, &dist);
    }

    last_regime_close_reason = reason ? reason : "";
    note_regime_close(action, delta);
    return action;
}

int WorkloadProfiler::size_area_index(double avg_size) {
    if (avg_size < 50.0) return 0;
    if (avg_size < 100.0) return 1;
    if (avg_size < 500.0) return 2;
    if (avg_size < 1000.0) return 3;
    if (avg_size < 2000.0) return 4;
    if (avg_size < 10000.0) return 5;
    if (avg_size < 50000.0) return 6;
    return 7;
}

const char* WorkloadProfiler::size_area_name(int idx) {
    static const char* kNames[N_SIZE_AREAS] = {
        "<50", "50-100", "100-500", "500-1000",
        "1000-2000", "2000-10000", "10000-50000", ">50000"
    };
    if (idx < 0 || idx >= N_SIZE_AREAS) return "ALL";
    return kNames[idx];
}

int WorkloadProfiler::match_band_index(bool usable, double distance) {
    if (!usable || !std::isfinite(distance)) return 6;
    // Nearest-neighbour sigma distance. No containment band.
    if (distance <= 0.5) return 0;
    if (distance <= 1.0) return 1;
    if (distance <= 1.5) return 2;
    if (distance <= 2.5) return 3;
    if (distance <= 4.0) return 4;
    return 5;
}

const char* WorkloadProfiler::match_band_name(int idx) {
    static const char* kNames[N_MATCH_BANDS] = {
        "<=0.5", "0.5-1", "1-1.5", "1.5-2.5",
        "2.5-4", ">4", "unmatched"
    };
    if (idx < 0 || idx >= N_MATCH_BANDS) return "unmatched";
    return kNames[idx];
}

void WorkloadProfiler::set_test_match_stats_paths(
        const std::string& events_path, const std::string& stats_path,
        const std::string& nn_path) {
    test_match_events_path = events_path;
    test_match_stats_path = stats_path;
    nn_distances_path = nn_path;
}

void WorkloadProfiler::set_test_type_stats_path(const std::string& stats_path) {
    test_type_stats_path = stats_path;
}

void WorkloadProfiler::set_window_start_audit(
        int start_arm, const std::vector<double>& start_arc) {
    active_policy_arm = start_arm;
    active_policy_weights = start_arc;
    active_inject_features = current_features;
    if (!active_policy_match) {
        active_policy_type = classify_regime_type(current_features);
    }
}

void WorkloadProfiler::note_test_match_close(double delta_bmr,
                                             uint64_t win_requests,
                                             uint64_t win_bytes,
                                             const char* close_reason) {
    if (win_requests == 0) return;
    TestMatchEvent ev;
    ev.avg_size = static_cast<double>(win_bytes) / win_requests;
    ev.len = win_requests;
    ev.bytes = win_bytes;
    ev.match_band = active_match_band;
    ev.nn_dist = active_nn_dist;
    ev.policy_applied = active_policy_match;
    ev.match_refused = active_match_refused;
    ev.distance =
        (active_policy_match || active_match_refused) ? active_match_dist : -1.0;
    ev.close_distance =
        (active_policy_row >= 0 &&
         active_policy_row < static_cast<int>(policy_matrix.size()))
        ? calculate_aggregate_distance(
              current_features,
              policy_matrix[static_cast<size_t>(active_policy_row)].features)
        : -1.0;
    ev.delta_bmr = delta_bmr;
    ev.type_idx = classify_regime_type(current_features);
    ev.inject_type_idx =
        (active_policy_match || active_match_refused)
        ? classify_regime_type(active_inject_features)
        : N_REGIME_TYPES - 1;
    ev.policy_type_idx = active_policy_type;
    ev.policy_row = active_policy_row;
    ev.policy_arm = active_policy_arm;
    ev.close_reason = close_reason ? close_reason : "unknown";
    ev.policy_weights = active_policy_weights;
    ev.inject_features = active_inject_features;
    ev.close_features = current_features;

    const int sidx = size_area_index(ev.avg_size);
    int band = ev.match_band;
    if (band < 0 || band >= N_MATCH_BANDS) band = N_MATCH_BANDS - 1;
    int type_idx = ev.type_idx;
    if (type_idx < 0 || type_idx >= N_REGIME_TYPES) type_idx = N_REGIME_TYPES - 1;
    auto add = [&](TestAgg& c) {
        c.regimes++;
        c.bytes += ev.bytes;
        if (ev.match_refused) {
            c.refused++;
            c.refused_bytes += ev.bytes;
        }
        if (ev.policy_applied) {
            c.applied++;
            c.arm_chosen++;
            c.applied_bytes += ev.bytes;
            c.delta_sum += ev.delta_bmr;
            c.byte_delta_sum += ev.delta_bmr * static_cast<double>(ev.bytes);
            sample_add(c.deltas, ev.delta_bmr);
            if (ev.delta_bmr > 0.0) c.won++;
            else if (ev.delta_bmr < 0.0) c.lost++;
        }
    };
    add(test_band[band][sidx]);
    add(test_band[band][N_SIZE_AREAS]);
    add(test_grand);
    add(test_type[type_idx][sidx]);
    add(test_type[type_idx][N_SIZE_AREAS]);
    add(test_type[N_REGIME_TYPES][sidx]);
    add(test_type[N_REGIME_TYPES][N_SIZE_AREAS]);

    if (!test_match_events_path.empty() || !nn_distances_path.empty()) {
        test_match_events.push_back(ev);
    }
    test_stat_closes++;
    if (test_match_events.size() >= 256) {
        flush_test_match_events(false);
    }
    if (test_stat_closes == 1 ||
        (test_stat_closes - test_stat_last_rewrite) >= STATS_REWRITE_EVERY) {
        test_stat_last_rewrite = test_stat_closes;
        flush_test_match_events(true);
    }
}

void WorkloadProfiler::flush_test_match_events(bool rebuild_stats) {
    if (!test_match_events.empty() && !test_match_events_path.empty()) {
        if (ensure_parent_dir(test_match_events_path)) {
            const bool exists =
                std::filesystem::exists(test_match_events_path);
            std::ofstream file(test_match_events_path, std::ios::app);
            if (file.is_open()) {
                if (!exists) {
                    file << "avg_size,len,bytes,nearest_band,policy_applied,"
                            "nn1,nn2,nn3,nn4,nn5,delta_bmr,type_idx,decision,"
                            "close_distance,close_reason,policy_row,policy_arm,"
                            "policy_type_idx,inject_type_idx,"
                            "w0,w1,w2,w3";
                    for (int i = 0; i < N_FEATURES; ++i) {
                        file << ",inject_" << feature_name(i);
                    }
                    for (int i = 0; i < N_FEATURES; ++i) {
                        file << ",close_" << feature_name(i);
                    }
                    file << "\n";
                }
                for (const auto& ev : test_match_events) {
                    const char* decision = ev.policy_applied
                        ? "applied"
                        : (ev.match_refused ? "refused_far" : "unmatched");
                    file << ev.avg_size << ","
                         << ev.len << ","
                         << ev.bytes << ","
                         << ev.match_band << ","
                         << (ev.policy_applied ? 1 : 0);
                    for (double d : ev.nn_dist) file << "," << d;
                    file << ","
                         << ev.delta_bmr << ","
                         << ev.type_idx << ","
                         << decision << ","
                         << ev.close_distance << ","
                         << ev.close_reason << ","
                         << ev.policy_row << ","
                         << ev.policy_arm << ","
                         << ev.policy_type_idx << ","
                         << ev.inject_type_idx;
                    for (int i = 0; i < 4; ++i) {
                        file << ","
                             << (i < static_cast<int>(ev.policy_weights.size())
                                 ? ev.policy_weights[static_cast<size_t>(i)] : 0.0);
                    }
                    file << ",";
                    write_features_csv(file, ev.inject_features);
                    file << ",";
                    write_features_csv(file, ev.close_features);
                    file << "\n";
                }
            }
        }
    }
    if (!test_match_events.empty() && !nn_distances_path.empty()) {
        if (ensure_parent_dir(nn_distances_path)) {
            const bool exists = std::filesystem::exists(nn_distances_path);
            std::ofstream file(nn_distances_path, std::ios::app);
            if (file.is_open()) {
                if (!exists) {
                    file << "avg_size,bytes,delta_bmr,nn1,nn2,nn3,nn4,nn5\n";
                }
                for (const auto& ev : test_match_events) {
                    file << ev.avg_size << ","
                         << ev.bytes << ","
                         << ev.delta_bmr;
                    for (double d : ev.nn_dist) file << "," << d;
                    file << "\n";
                }
            }
        }
    }
    if (!test_match_events.empty()) {
        test_match_events.clear();
    }
    if (!rebuild_stats) return;
    rewrite_test_match_stats();
    rewrite_test_type_stats();
}

void WorkloadProfiler::rewrite_test_match_stats() const {
    if (test_match_stats_path.empty()) return;
    if (!ensure_parent_dir(test_match_stats_path)) return;

    auto median_d = [](std::vector<double> v) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        const size_t n = v.size();
        return (n % 2 == 1)
            ? v[n / 2]
            : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    };

    std::string tmp = test_match_stats_path + ".tmp";
    std::ofstream out(tmp, std::ios::trunc);
    if (!out.is_open()) return;
    out << "nearest_band,size_area,regimes,applied,refused,apply_pct,"
           "applied_bytes,refused_bytes,apply_bytes_pct,"
           "won,lost,win_pct,median_bmr_saved,avg_bmr_saved,"
           "byte_weighted_bmr_saved,share_of_bytes_pct\n";

    const double total_bytes = static_cast<double>(test_grand.bytes);
    auto write_cell = [&](const char* band_name, const char* area_name,
                          const TestAgg& c) {
        const double win_pct = (c.applied > 0)
            ? 100.0 * static_cast<double>(c.won) / c.applied : 0.0;
        const double apply_pct = (c.regimes > 0)
            ? 100.0 * static_cast<double>(c.applied) / c.regimes : 0.0;
        const double apply_bytes_pct = (c.bytes > 0)
            ? 100.0 * static_cast<double>(c.applied_bytes) / c.bytes : 0.0;
        const double avg_delta = (c.applied > 0)
            ? c.delta_sum / static_cast<double>(c.applied) : 0.0;
        const double byte_delta = (c.applied_bytes > 0)
            ? c.byte_delta_sum / static_cast<double>(c.applied_bytes) : 0.0;
        const double share = (total_bytes > 0.0)
            ? 100.0 * static_cast<double>(c.bytes) / total_bytes : 0.0;
        out << band_name << ","
            << area_name << ","
            << c.regimes << ","
            << c.applied << ","
            << c.refused << ","
            << apply_pct << ","
            << c.applied_bytes << ","
            << c.refused_bytes << ","
            << apply_bytes_pct << ","
            << c.won << ","
            << c.lost << ","
            << win_pct << ","
            << median_d(c.deltas.v) << ","
            << avg_delta << ","
            << byte_delta << ","
            << share << "\n";
    };

    for (int band = 0; band < N_MATCH_BANDS; ++band) {
        for (int size = 0; size < N_SIZE_AREAS; ++size) {
            write_cell(match_band_name(band), size_area_name(size),
                       test_band[band][size]);
        }
        write_cell(match_band_name(band), "ALL",
                   test_band[band][N_SIZE_AREAS]);
    }
    write_cell("ALL", "ALL", test_grand);
    out.close();

    std::error_code ec;
    std::filesystem::rename(tmp, test_match_stats_path, ec);
    if (ec) {
        std::ifstream src(tmp, std::ios::binary);
        std::ofstream dst(test_match_stats_path,
                          std::ios::binary | std::ios::trunc);
        if (src && dst) dst << src.rdbuf();
        std::filesystem::remove(tmp, ec);
    }
}

void WorkloadProfiler::rewrite_test_type_stats() const {
    if (test_type_stats_path.empty()) return;
    if (!ensure_parent_dir(test_type_stats_path)) return;

    auto median_d = [](std::vector<double> v) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        size_t n = v.size();
        if (n % 2 == 1) return v[n / 2];
        return 0.5 * (v[n / 2 - 1] + v[n / 2]);
    };

    const double total_bytes =
        static_cast<double>(test_type[N_REGIME_TYPES][N_SIZE_AREAS].bytes);

    std::string tmp = test_type_stats_path + ".tmp";
    std::ofstream out(tmp, std::ios::trunc);
    if (!out.is_open()) return;
    out << "type,area,regimes,won,lost,win_pct,median_bmr_saved,avg_bmr_saved,"
           "share_of_bytes_pct,arm_chosen,refused\n";

    auto write_cell = [&](const char* type_name, const char* area_name,
                          const TestAgg& c) {
        const double win_pct = (c.arm_chosen > 0)
            ? (100.0 * static_cast<double>(c.won) / c.arm_chosen) : 0.0;
        const double avg_delta = (c.arm_chosen > 0)
            ? (c.delta_sum / static_cast<double>(c.arm_chosen)) : 0.0;
        const double share = (total_bytes > 0.0)
            ? (100.0 * static_cast<double>(c.bytes) / total_bytes) : 0.0;
        out << type_name << ","
            << area_name << ","
            << c.regimes << ","
            << c.won << ","
            << c.lost << ","
            << win_pct << ","
            << median_d(c.deltas.v) << ","
            << avg_delta << ","
            << share << ","
            << c.arm_chosen << ","
            << c.refused << "\n";
    };

    for (int t = 0; t < N_REGIME_TYPES; ++t) {
        for (int s = 0; s < N_SIZE_AREAS; ++s) {
            write_cell(regime_type_name(t), size_area_name(s), test_type[t][s]);
        }
        write_cell(regime_type_name(t), "ALL", test_type[t][N_SIZE_AREAS]);
    }
    for (int s = 0; s < N_SIZE_AREAS; ++s) {
        write_cell("ALL", size_area_name(s), test_type[N_REGIME_TYPES][s]);
    }
    write_cell("ALL", "ALL", test_type[N_REGIME_TYPES][N_SIZE_AREAS]);
    out.close();

    std::error_code ec;
    std::filesystem::rename(tmp, test_type_stats_path, ec);
    if (ec) {
        std::ifstream src(tmp, std::ios::binary);
        std::ofstream dst(test_type_stats_path,
                          std::ios::binary | std::ios::trunc);
        if (src && dst) dst << src.rdbuf();
        std::filesystem::remove(tmp, ec);
    }
}

void WorkloadProfiler::sample_add(SampleSet& s, double x) {
    s.seen++;
    if (s.v.size() < STATS_SAMPLE_CAP) {
        s.v.push_back(x);
        return;
    }
    stats_rng ^= stats_rng << 13;
    stats_rng ^= stats_rng >> 7;
    stats_rng ^= stats_rng << 17;
    const uint64_t j = stats_rng % s.seen;
    if (j < STATS_SAMPLE_CAP) s.v[static_cast<size_t>(j)] = x;
}

void WorkloadProfiler::note_regime_close(const char* action, double delta_bmr) {
    if (phase_requests == 0) return;
    RegimeCloseEvent ev;
    ev.avg_size = static_cast<double>(phase_bytes) / phase_requests;
    // Length for stats stays full-regime; store quality uses committed window.
    ev.len = phase_requests;
    ev.bytes = phase_bytes;
    // "long_enough" now means "had a committed scoring window at all".
    ev.long_enough = (phase_score_active && score_bytes > 0);
    ev.arm_chosen = ev.long_enough && !last_stable_weights.empty();
    ev.delta_bmr = delta_bmr;
    const std::string a = action ? action : "";
    ev.stored = (a == "upsert_new");
    // Why a regime did or did not earn a row. Without this the only visible
    // signal is the final row count, which cannot distinguish "never scored"
    // from "merged into an existing box".
    action_counts[a]++;
    ev.type_idx = classify_regime_type(current_features);

    // Live aggregates first: the summaries must not depend on the dumps.
    const int sidx = size_area_index(ev.avg_size);
    auto accumulate = [&](TrainAgg& a, bool with_lens) {
        a.regimes++;
        a.bytes += ev.bytes;
        if (with_lens) sample_add(a.lens, static_cast<double>(ev.len));
        if (ev.long_enough) a.long_enough++;
        if (ev.arm_chosen) {
            a.arm_chosen++;
            a.delta_n++;
            a.delta_sum += ev.delta_bmr;
            sample_add(a.deltas, ev.delta_bmr);
            if (ev.delta_bmr > 0.0) a.beat++;
            else if (ev.delta_bmr < 0.0) a.lost++;
        }
        if (ev.stored) a.stored++;
        // Floor means the regime closed on the first tick the minimum
        // allowed. Drift and max-length are the two clock reasons.
        if (ev.len < MIN_REGIME_LEN + REGIME_TICK) a.at_min_len++;
        if (last_regime_close_reason == "feature_drift") a.drift_close++;
        else if (last_regime_close_reason == "max_len") a.max_len_close++;
    };
    accumulate(train_band[sidx], true);
    accumulate(train_all, true);
    // The arm that played this regime, scored on the bytes that produced
    // the delta. Feature type is the row; size is not.
    if (ev.arm_chosen &&
        active_train_arm >= 0 && active_train_arm < N_TRAIN_ARMS &&
        ev.type_idx >= 0 && ev.type_idx < N_REGIME_TYPES) {
        const uint64_t scored_bytes = score_bytes;
        auto add_arm = [&](ArmTypeAgg& cell) {
            cell.regimes++;
            cell.bytes += scored_bytes;
            cell.byte_delta_sum +=
                ev.delta_bmr * static_cast<double>(scored_bytes);
            if (ev.delta_bmr > 0.0) {
                cell.won++;
                cell.win_delta_sum += ev.delta_bmr;
            } else if (ev.delta_bmr < 0.0) {
                cell.lost++;
                cell.loss_delta_sum += ev.delta_bmr;
            } else {
                cell.draw++;
            }
        };
        add_arm(arm_type[active_train_arm][ev.type_idx]);
        add_arm(arm_type[active_train_arm][N_REGIME_TYPES]);
    }

    // Reservoir of feature vectors for the p5/p95 columns.
    feature_reservoir_seen++;
    if (feature_reservoir.size() < STATS_SAMPLE_CAP) {
        feature_reservoir.push_back(to_vector(current_features));
    } else {
        stats_rng ^= stats_rng << 13;
        stats_rng ^= stats_rng >> 7;
        stats_rng ^= stats_rng << 17;
        const uint64_t j = stats_rng % feature_reservoir_seen;
        if (j < STATS_SAMPLE_CAP) {
            feature_reservoir[static_cast<size_t>(j)] =
                to_vector(current_features);
        }
    }

    if (!regime_events_path.empty()) regime_events.push_back(ev);
    if (!feature_samples_path.empty()) {
        feature_samples.push_back(to_vector(current_features));
    }

    stat_closes++;
    if (regime_events.size() >= 256) {
        flush_regime_events(false);
    }
    // Dumps may be off, so never wait on regime_events.size() to write stats.
    if (stat_closes == 1 ||
        (stat_closes - stat_last_rewrite) >= STATS_REWRITE_EVERY) {
        stat_last_rewrite = stat_closes;
        flush_regime_events(true);
    }
}

void WorkloadProfiler::set_regime_stats_paths(const std::string& events_path,
                                              const std::string& stats_path) {
    regime_events_path = events_path;
    regime_stats_path = stats_path;
}

void WorkloadProfiler::set_feature_stats_paths(const std::string& samples_path,
                                               const std::string& stats_path) {
    feature_samples_path = samples_path;
    feature_stats_path = stats_path;
}

void WorkloadProfiler::set_type_stats_path(const std::string& stats_path) {
    type_stats_path = stats_path;
}

const char* WorkloadProfiler::regime_type_name(int idx) {
    static const char* kNames[N_REGIME_TYPES] = {
        "scan_heavy",
        "reuse_heavy",
        "skewed_hotspots",
        "flat_popularity",
        "sequential",
        "random_access",
        "long_reuse",
        "short_reuse",
        "bursty",
        "steady",
        "mixed"
    };
    if (idx < 0 || idx >= N_REGIME_TYPES) return "mixed";
    return kNames[idx];
}

int WorkloadProfiler::classify_regime_type(const WorkloadFeatures& f) {
    // Score each named type; pick the strongest if above a floor, else mixed.
    // Request size is intentionally NOT used here (it is the other table axis).
    double s[N_REGIME_TYPES];
    for (int i = 0; i < N_REGIME_TYPES; ++i) s[i] = 0.0;

    // 0 scan_heavy
    s[0] = 0.55 * f.scan_ratio + 0.45 * f.singleton_ratio;
    if (f.scan_ratio < 0.35) s[0] *= 0.25;

    // 1 reuse_heavy
    s[1] = 0.55 * std::min(1.0, f.avg_frequency / 6.0)
         + 0.45 * (1.0 - std::min(1.0, f.object_diversity));
    if (f.avg_frequency < 2.0) s[1] *= 0.3;

    // 2 skewed_hotspots
    s[2] = 0.7 * std::min(1.0, f.popularity_skewness / 8.0)
         + 0.3 * std::min(1.0, f.avg_frequency / 5.0);
    if (f.popularity_skewness < 2.0) s[2] *= 0.25;

    // 3 flat_popularity
    s[3] = 0.5 * (1.0 - std::min(1.0, f.popularity_skewness / 4.0))
         + 0.3 * std::min(1.0, f.object_diversity)
         + 0.2 * std::min(1.0, std::max(0.0, f.avg_frequency - 1.0) / 3.0);
    if (f.popularity_skewness > 2.5) s[3] *= 0.2;

    // 4 sequential
    s[4] = f.sequentiality_ratio;
    if (f.sequentiality_ratio < 0.25) s[4] = 0.0;

    // 5 random_access
    s[5] = (1.0 - std::min(1.0, f.sequentiality_ratio * 4.0))
         * (1.0 - std::min(1.0, f.scan_ratio));
    if (f.sequentiality_ratio > 0.15 || f.scan_ratio > 0.55) s[5] *= 0.2;

    // 6 long_reuse
    s[6] = 0.6 * std::min(1.0, f.avg_reuse_distance / 50000.0)
         + 0.4 * std::min(1.0, f.avg_frequency / 4.0);
    if (f.avg_reuse_distance < 5000.0 || f.avg_frequency < 1.5) s[6] *= 0.25;

    // 7 short_reuse
    s[7] = 0.5 * (1.0 - std::min(1.0, f.avg_reuse_distance / 2000.0))
         + 0.5 * std::min(1.0, f.avg_frequency / 5.0);
    if (f.avg_reuse_distance > 3000.0 || f.avg_frequency < 2.0) s[7] *= 0.25;

    // 8 bursty
    s[8] = 0.6 * std::min(1.0, f.burstiness_index / 3.0)
         + 0.4 * std::min(1.0, f.arrival_time_variance / 4.0);
    if (f.burstiness_index < 1.0) s[8] *= 0.25;

    // 9 steady
    s[9] = 0.6 * (1.0 - std::min(1.0, f.burstiness_index / 1.5))
         + 0.4 * (1.0 - std::min(1.0, f.arrival_time_variance / 2.0));
    if (f.burstiness_index > 1.2) s[9] *= 0.2;

    // 10 mixed is fallback only
    s[10] = 0.0;

    int best = 10;
    double best_s = 0.34; // floor: below this → mixed
    for (int i = 0; i < 10; ++i) {
        if (s[i] > best_s) {
            best_s = s[i];
            best = i;
        }
    }
    return best;
}

const char* WorkloadProfiler::feature_name(int idx) {
    static const char* kNames[N_FEATURES] = {
        "write_ratio", "avg_request_size", "size_variance", "singleton_ratio",
        "popularity_skewness", "avg_frequency", "object_diversity",
        "avg_reuse_distance", "sequentiality_ratio", "out_cache_hit_rate",
        "request_rate", "burstiness_index", "arrival_time_variance",
        "working_set_byte_delta", "scan_ratio"
    };
    if (idx < 0 || idx >= N_FEATURES) return "unknown";
    return kNames[idx];
}

void WorkloadProfiler::flush_feature_samples() {
    if (feature_samples.empty() || feature_samples_path.empty()) return;
    if (!ensure_parent_dir(feature_samples_path)) return;

    const bool exists = std::filesystem::exists(feature_samples_path);
    std::ofstream file(feature_samples_path, std::ios::app);
    if (!file.is_open()) return;
    if (!exists) {
        for (int i = 0; i < N_FEATURES; ++i) {
            if (i) file << ",";
            file << feature_name(i);
        }
        file << "\n";
    }
    for (const auto& v : feature_samples) {
        for (int i = 0; i < N_FEATURES; ++i) {
            if (i) file << ",";
            file << ((i < (int)v.size()) ? v[i] : 0.0);
        }
        file << "\n";
    }
    file.close();
    feature_samples.clear();
}

void WorkloadProfiler::flush_regime_events(bool rebuild_stats) {
    flush_feature_samples();
    if (!regime_events.empty() && !regime_events_path.empty()) {
        if (ensure_parent_dir(regime_events_path)) {
            const bool exists = std::filesystem::exists(regime_events_path);
            std::ofstream file(regime_events_path, std::ios::app);
            if (file.is_open()) {
                if (!exists) {
                    file << "avg_size,len,bytes,long_enough,arm_chosen,delta_bmr,stored,type_idx\n";
                }
                for (const auto& ev : regime_events) {
                    file << ev.avg_size << ","
                         << ev.len << ","
                         << ev.bytes << ","
                         << (ev.long_enough ? 1 : 0) << ","
                         << (ev.arm_chosen ? 1 : 0) << ","
                         << ev.delta_bmr << ","
                         << (ev.stored ? 1 : 0) << ","
                         << ev.type_idx << "\n";
                }
            }
        }
        regime_events.clear();
    }
    if (!rebuild_stats) return;
    rewrite_regime_stats();
    rewrite_type_stats();
    if (stat_closes == 1 ||
        (stat_closes - feature_stat_last_rewrite) >= FEATURE_STATS_EVERY) {
        feature_stat_last_rewrite = stat_closes;
        rewrite_feature_stats();
    }
}

void WorkloadProfiler::rewrite_regime_stats() const {
    if (regime_stats_path.empty()) return;
    if (!ensure_parent_dir(regime_stats_path)) return;

    const TrainAgg* bands = train_band;
    const TrainAgg& all = train_all;

    auto median_u = [](std::vector<double> v) -> uint64_t {
        if (v.empty()) return 0;
        std::sort(v.begin(), v.end());
        return static_cast<uint64_t>(v[v.size() / 2]);
    };

    std::string tmp = regime_stats_path + ".tmp";
    std::ofstream out(tmp, std::ios::trunc);
    if (!out.is_open()) return;
    // Share of bytes, how many regimes closed, and how many of those regimes
    // beat the shadow. The last four columns say whether the clock is cutting
    // at the floor, on drift, or at the maximum length.
    out << "area,share_of_bytes_pct,regimes,beat_3l_pct,median_requests,"
           "at_min_len_pct,drift_close_pct,max_len_close_pct\n";

    const double total_bytes = static_cast<double>(all.bytes);
    auto write_band = [&](const char* name, const TrainAgg& b) {
        const double share = (total_bytes > 0.0)
            ? (100.0 * static_cast<double>(b.bytes) / total_bytes) : 0.0;
        const double beat_pct = (b.regimes > 0)
            ? (100.0 * static_cast<double>(b.beat) / b.regimes) : 0.0;
        auto pct = [&](uint64_t n) -> double {
            return (b.regimes > 0)
                ? (100.0 * static_cast<double>(n) / b.regimes) : 0.0;
        };
        out << name << ","
            << share << ","
            << b.regimes << ","
            << beat_pct << ","
            << median_u(b.lens.v) << ","
            << pct(b.at_min_len) << ","
            << pct(b.drift_close) << ","
            << pct(b.max_len_close) << "\n";
    };

    for (int i = 0; i < N_SIZE_AREAS; ++i) {
        write_band(size_area_name(i), bands[i]);
    }
    write_band("ALL", all);
    out.close();
    std::error_code ec;
    std::filesystem::rename(tmp, regime_stats_path, ec);
    if (ec) {
        std::ifstream src(tmp, std::ios::binary);
        std::ofstream dst(regime_stats_path, std::ios::binary | std::ios::trunc);
        if (src && dst) dst << src.rdbuf();
        std::filesystem::remove(tmp, ec);
    }
}

const char* WorkloadProfiler::train_arm_name(int arm) {
    static const char* kNames[N_TRAIN_ARMS] = {
        "Balanced ends",
        "Tail+",
        "Head+",
        "Explore",
        "Lean middle",
        "Lean ends"
    };
    if (arm < 0 || arm >= N_TRAIN_ARMS) return "unknown";
    return kNames[arm];
}

void WorkloadProfiler::rewrite_type_stats() const {
    if (type_stats_path.empty()) return;
    if (!ensure_parent_dir(type_stats_path)) return;

    // One row per arm × feature type, then one ALL row per arm.
    // Zero rows stay, so a type an arm never played is visible.
    const int n_arms = std::min(N_TRAIN_ARMS,
                                std::max(1, static_cast<int>(current_mab_k)));

    std::string tmp = type_stats_path + ".tmp";
    std::ofstream out(tmp, std::ios::trunc);
    if (!out.is_open()) return;
    out << "arm,type,regimes,arm_share_of_type_pct,win_pct,draw_pct,loss_pct,"
           "mean_win_delta,mean_loss_delta,byte_weighted_mean_delta\n";

    auto write_cell = [&](int arm, int type_idx) {
        const ArmTypeAgg& c = arm_type[arm][type_idx];
        uint64_t type_regimes = 0;
        for (int a = 0; a < n_arms; ++a) {
            type_regimes += arm_type[a][type_idx].regimes;
        }
        const double share = (type_regimes > 0)
            ? (100.0 * static_cast<double>(c.regimes) / type_regimes) : 0.0;
        const double win_pct = (c.regimes > 0)
            ? (100.0 * static_cast<double>(c.won) / c.regimes) : 0.0;
        const double draw_pct = (c.regimes > 0)
            ? (100.0 * static_cast<double>(c.draw) / c.regimes) : 0.0;
        const double loss_pct = (c.regimes > 0)
            ? (100.0 * static_cast<double>(c.lost) / c.regimes) : 0.0;
        const double mean_win = (c.won > 0)
            ? (c.win_delta_sum / static_cast<double>(c.won)) : 0.0;
        const double mean_loss = (c.lost > 0)
            ? (c.loss_delta_sum / static_cast<double>(c.lost)) : 0.0;
        const double byte_mean = (c.bytes > 0)
            ? (c.byte_delta_sum / static_cast<double>(c.bytes)) : 0.0;
        const char* type_name = (type_idx == N_REGIME_TYPES)
            ? "ALL" : regime_type_name(type_idx);
        out << train_arm_name(arm) << ","
            << type_name << ","
            << c.regimes << ","
            << share << ","
            << win_pct << ","
            << draw_pct << ","
            << loss_pct << ","
            << mean_win << ","
            << mean_loss << ","
            << byte_mean << "\n";
    };

    for (int t = 0; t < N_REGIME_TYPES; ++t) {
        for (int arm = 0; arm < n_arms; ++arm) write_cell(arm, t);
    }
    for (int arm = 0; arm < n_arms; ++arm) write_cell(arm, N_REGIME_TYPES);

    out.close();
    std::error_code ec;
    std::filesystem::rename(tmp, type_stats_path, ec);
    if (ec) {
        std::ifstream src(tmp, std::ios::binary);
        std::ofstream dst(type_stats_path, std::ios::binary | std::ios::trunc);
        if (src && dst) dst << src.rdbuf();
        std::filesystem::remove(tmp, ec);
    }
}

void WorkloadProfiler::rewrite_feature_stats() const {
    if (feature_stats_path.empty()) return;
    if (!ensure_parent_dir(feature_stats_path)) return;

    // Reservoir of per-regime feature vectors, kept in memory.
    const std::vector<std::vector<double>>& samples = feature_reservoir;

    // Table boxes from in-memory policy matrix (same process as TRAIN upserts).
    std::vector<std::pair<std::vector<double>, std::vector<double>>> boxes;
    for (const auto& rec : policy_matrix) {
        if (!rec.has_box) continue;
        boxes.push_back({to_vector(rec.feat_min), to_vector(rec.feat_max)});
    }

    auto quantile = [](std::vector<double> v, double q) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        if (v.size() == 1) return v[0];
        double pos = q * static_cast<double>(v.size() - 1);
        size_t i = static_cast<size_t>(pos);
        double frac = pos - static_cast<double>(i);
        if (i + 1 >= v.size()) return v.back();
        return v[i] * (1.0 - frac) + v[i + 1] * frac;
    };

    auto union_len = [](std::vector<std::pair<double, double>> iv) -> double {
        if (iv.empty()) return 0.0;
        std::sort(iv.begin(), iv.end());
        double total = 0.0;
        double a = iv[0].first, b = iv[0].second;
        for (size_t i = 1; i < iv.size(); ++i) {
            if (iv[i].first <= b) {
                b = std::max(b, iv[i].second);
            } else {
                total += std::max(0.0, b - a);
                a = iv[i].first;
                b = iv[i].second;
            }
        }
        total += std::max(0.0, b - a);
        return total;
    };

    std::string tmp = feature_stats_path + ".tmp";
    std::ofstream out(tmp, std::ios::trunc);
    if (!out.is_open()) return;
    // Only features the mask turned on. Cover of an off feature is the
    // constant zero overlapping itself, which reads as healthy and is not.
    // spread is p95-p5. box_over_spread is how many times wider the stored
    // extremes are than that mass. flat: the feature does not move.
    // weak: the boxes miss the mass. wide: extremes dwarf the mass.
    out << "feature,role,spread,box_over_spread,cover_pct,verdict\n";

    for (int i = 0; i < N_FEATURES; ++i) {
        if (!feature_active[i]) continue;
        const bool usable = feature_usable(static_cast<size_t>(i));
        const bool opens = feature_matchable(static_cast<size_t>(i));
        std::vector<double> col;
        col.reserve(samples.size());
        for (const auto& s : samples) {
            if (i < (int)s.size()) col.push_back(s[i]);
        }

        const double p5 = quantile(col, 0.05);
        const double p95 = quantile(col, 0.95);

        double table_lo = 0.0, table_hi = 0.0;
        bool have_box = false;
        std::vector<std::pair<double, double>> clipped;
        int boxes_touch = 0;
        for (const auto& bx : boxes) {
            if (i >= (int)bx.first.size() || i >= (int)bx.second.size()) continue;
            double lo = bx.first[i];
            double hi = bx.second[i];
            if (hi < lo) std::swap(lo, hi);
            if (!have_box) {
                table_lo = lo;
                table_hi = hi;
                have_box = true;
            } else {
                table_lo = std::min(table_lo, lo);
                table_hi = std::max(table_hi, hi);
            }
            // Overlap with mass [p5, p95]
            double mass_lo = p5, mass_hi = p95;
            if (mass_hi < mass_lo) std::swap(mass_lo, mass_hi);
            double clo = std::max(lo, mass_lo);
            double chi = std::min(hi, mass_hi);
            if (chi >= clo) {
                boxes_touch++;
                clipped.push_back({clo, chi});
            }
        }

        double cover = 0.0;
        const double mass = std::max(0.0, p95 - p5);
        if (mass <= 1e-15) {
            // Degenerate mass: covered if any box exists at that constant value.
            cover = (boxes_touch > 0) ? 100.0 : 0.0;
        } else {
            cover = 100.0 * std::min(1.0, union_len(clipped) / mass);
        }

        const double spread = std::abs(p95 - p5);
        const double box_span = have_box ? std::abs(table_hi - table_lo) : 0.0;
        const double box_over = (spread > 1e-12) ? (box_span / spread) : 0.0;
        const char* role = !usable ? "unused"
                          : (opens ? "opens rows" : "distance only");
        const char* verdict = "good";
        if (spread <= 1e-12) verdict = "flat";
        else if (cover < 5.0) verdict = "weak";
        else if (box_over >= 4.0) verdict = "wide";

        out << feature_name(i) << ","
            << role << ","
            << spread << ","
            << box_over << ","
            << cover << ","
            << verdict << "\n";
    }

    out.close();

    std::error_code ec;
    std::filesystem::rename(tmp, feature_stats_path, ec);
    if (ec) {
        std::ifstream src(tmp, std::ios::binary);
        std::ofstream dst(feature_stats_path, std::ios::binary | std::ios::trunc);
        if (src && dst) dst << src.rdbuf();
        std::filesystem::remove(tmp, ec);
    }
}

void WorkloadProfiler::seed_initial_policies() {
    // No synthetic policies: each v4 row must come from a measured regime.
}

void WorkloadProfiler::init(const std::string& mode_str, 
                            const std::string& policy_file, 
                            const std::string& config_file, 
                            size_t cache_size,
                            uint8_t mab_k,
                            bool train_log_enable,
                            const std::string& train_log_prefix,
                            uint64_t train_log_interval,
                            const std::string& feature_mask,
                            bool persist_event_logs) {
    if (feature_mask.size() != N_FEATURES ||
        feature_mask.find_first_not_of("01") != std::string::npos) {
        throw std::runtime_error(
            "feature-mask must contain exactly 15 characters of 0 or 1");
    }
    int active_count = 0;
    for (int i = 0; i < N_FEATURES; ++i) {
        feature_active[i] = feature_mask[static_cast<size_t>(i)] == '1';
        if (feature_active[i]) active_count++;
    }
    if (active_count < 3) {
        throw std::runtime_error("feature-mask must enable at least 3 features");
    }
    feature_mask_str = feature_mask;
    rebuild_active_index();
    current_mab_k = mab_k;
    policy_filename = policy_file;
    config_filename = config_file;
    train_total_reqs = 0;
    train_total_misses = 0;
    phase_box_init = false;
    last_match_attempt_seq = 0;
    active_policy_match = false;
    feature_scale_loaded = false;

    if (!policy_filename.empty()) ensure_parent_dir(policy_filename);
    if (!config_filename.empty()) ensure_parent_dir(config_filename);
    
    size_t estimated_entries = cache_size;
    if (cache_size > 100000) { 
        estimated_entries = std::max((size_t)1, cache_size / 4096);
    }
    window_size = std::max(estimated_entries * 2, (size_t)10000);

    if (mode_str == "test") {
        current_mode = ProfilerMode::TEST;
        load_config();
        load_policy_matrix();
        train_log.init(false, train_log_prefix, train_log_interval, mab_k);
        std::filesystem::path prefix_path(train_log_prefix);
        std::filesystem::path log_dir = prefix_path.parent_path();
        if (log_dir.empty()) log_dir = ".";
        std::filesystem::path test_dir = log_dir / "test";
        set_test_match_stats_paths(
            persist_event_logs ? (test_dir / "match_events.csv").string()
                               : std::string(),
            (test_dir / "match_stats.csv").string(),
            (test_dir / "nn_distances.csv").string());
        set_test_type_stats_path((test_dir / "type_stats.csv").string());
    } else { 
        current_mode = ProfilerMode::TRAIN;
        load_config();
        load_policy_matrix();
        train_log.init(train_log_enable, train_log_prefix, train_log_interval, mab_k);
        // Regime + feature stats live under <log-dir>/train/ (TRAIN only).
        // train_log CSV files stay disabled unless explicitly enabled.
        {
            std::filesystem::path prefix_path(train_log_prefix);
            std::filesystem::path log_dir = prefix_path.parent_path();
            if (log_dir.empty()) log_dir = ".";
            std::filesystem::path train_dir = log_dir / "train";
            // Summaries always. The per-regime dumps are only a debugging
            // aid now that the summaries come from live counters, and they
            // are by far the largest files the run produces.
            set_regime_stats_paths(
                persist_event_logs
                    ? (train_dir / "regime_events.csv").string()
                    : std::string(),
                (train_dir / "regime_stats.csv").string());
            set_feature_stats_paths(
                persist_event_logs
                    ? (train_dir / "feature_samples.csv").string()
                    : std::string(),
                (train_dir / "feature_stats.csv").string());
            set_type_stats_path((train_dir / "type_stats.csv").string());
        }
        if (train_log.enabled()) {
            train_log.log_learning(0, "train_start", policy_matrix.size(), 1.0, 0.0, -1,
                                   std::vector<double>(mab_k, 1.0 / std::max(1, (int)mab_k)));
        }
    }
}

void WorkloadProfiler::rebuild_active_index() {
    active_idx.clear();
    for (size_t i = 0; i < 15; ++i) {
        if (feature_active[i]) active_idx.push_back(i);
    }
    need_reuse_distance = feature_active[7];
    need_working_set = feature_active[13];
    need_time_features =
        feature_active[10] || feature_active[11] || feature_active[12];
}

void WorkloadProfiler::scale_reservoir_add(const std::vector<double>& v) {
    scale_seen++;
    if (scale_reservoir.size() < SCALE_RESERVOIR_CAP) {
        scale_reservoir.push_back(v);
        return;
    }
    stats_rng ^= stats_rng << 13;
    stats_rng ^= stats_rng >> 7;
    stats_rng ^= stats_rng << 17;
    const uint64_t j = stats_rng % scale_seen;
    if (j < SCALE_RESERVOIR_CAP) {
        scale_reservoir[static_cast<size_t>(j)] = v;
    }
}

// Sigma from the spread of tick samples rather than from consecutive-sample
// deltas. Two neighbouring samples of a rolling-window feature are nearly
// identical no matter how far apart they are taken, so a delta-based
// estimator always collapses; a quantile spread over the whole trace does not.
void WorkloadProfiler::refresh_feature_scale() {
    if (scale_reservoir.size() < SCALE_MIN_SAMPLES) return;
    std::vector<double> col;
    col.reserve(scale_reservoir.size());
    for (size_t f : active_idx) {
        col.clear();
        for (const auto& s : scale_reservoir) {
            if (f < s.size() && std::isfinite(s[f])) col.push_back(s[f]);
        }
        if (col.size() < SCALE_MIN_SAMPLES) continue;
        scale_ready = true;
        std::sort(col.begin(), col.end());
        const size_t n = col.size();
        const double p16 = col[static_cast<size_t>(0.16 * (n - 1))];
        const double p50 = col[n / 2];
        const double p84 = col[static_cast<size_t>(0.84 * (n - 1))];
        feature_means[f] = p50;
        const double spread = 0.5 * (p84 - p16);
        feature_scale[f] = std::max(std::isfinite(spread) ? spread : 0.0,
                                    loaded_scale[f]);
        // Keep the v2 config field consistent for anything still reading it.
        feature_vars[f] = feature_scale[f] * feature_scale[f];
    }
}

double WorkloadProfiler::feature_std(size_t i) const {
    if (scale_ready || feature_scale_loaded) {
        const double s = feature_scale[i];
        if (std::isfinite(s) && s > 0.0) return s;
        const double rel = std::abs(feature_means[i]) * FEATURE_STD_REL_FLOOR;
        return std::max(rel, std::sqrt(FEATURE_VAR_FLOOR));
    }
    // Bootstrap before the reservoir has enough samples.
    const double var = (std::isfinite(feature_vars[i]) && feature_vars[i] > 0.0)
        ? feature_vars[i] : FEATURE_VAR_FLOOR;
    const double sd = std::sqrt(std::max(var, FEATURE_VAR_FLOOR));
    const double rel = std::abs(feature_means[i]) * FEATURE_STD_REL_FLOOR;
    return std::max(sd, rel);
}

bool WorkloadProfiler::feature_usable(size_t i) const {
    if (!feature_active[i]) return false;
    // Usable means "this feature actually moves across the workload". The
    // spread is measured over the whole trace, so a feature is only dropped
    // when it is genuinely constant (write_ratio on a read-only trace), not
    // merely slow-moving, which is what the old per-request EWMA punished.
    if (!scale_ready && !feature_scale_loaded) return true;
    const double eps =
        std::max(std::abs(feature_means[i]) * FEATURE_SPREAD_REL_EPS, 1e-12);
    return feature_scale[i] > eps;
}

bool WorkloadProfiler::feature_matchable(size_t i) const {
    return MATCH_STRUCTURAL[i] && feature_usable(i);
}

namespace {
constexpr int kIndexAxes[3] = {1, 7, 4}; // avg_request_size, reuse, skew
constexpr int kIndexBins = 16;
constexpr int kIndexMaxCells = 64;
}  // namespace

int WorkloadProfiler::index_bin(size_t feat, double x) const {
    if (!feature_matchable(feat) || !std::isfinite(x)) return 0;
    int b = 0;
    if (feat == 1) {
        b = static_cast<int>(std::floor(std::log2(std::max(x, 16.0)))) - 4;
    } else if (feat == 7) {
        b = static_cast<int>(std::floor(std::log2(std::max(x, 1.0))));
    } else {
        b = static_cast<int>(std::floor(x));
    }
    if (b < 0) return 0;
    if (b >= kIndexBins) return kIndexBins - 1;
    return b;
}

uint64_t WorkloadProfiler::index_pack(int a, int b, int c) {
    auto u = [](int x) -> uint64_t {
        return static_cast<uint64_t>(x + 0x100000) & 0x1FFFFFu;
    };
    return u(a) | (u(b) << 21) | (u(c) << 42);
}

bool WorkloadProfiler::index_linear_faster(IndexUse use) const {
    // A missed row in the coverage test inserts a duplicate and the table
    // never settles, so that scan stays exhaustive in TRAIN. It runs only on
    // a stored close. Regime-open lookup runs on every regime and may use the
    // grid in either mode.
    if (use == IndexUse::Cover && current_mode == ProfilerMode::TRAIN) {
        return true;
    }
    const size_t n = policy_matrix.size();
    if (n <= 48) return true;
    return policy_wide.size() * 4 >= n;
}

bool WorkloadProfiler::index_box_too_wide(const WorkloadFeatures& box_min,
                                          const WorkloadFeatures& box_max) const {
    const auto mn = to_vector(box_min);
    const auto mx = to_vector(box_max);
    int cells = 1;
    for (int k = 0; k < 3; ++k) {
        const size_t f = static_cast<size_t>(kIndexAxes[k]);
        if (!feature_matchable(f)) continue;
        int lo = index_bin(f, mn[f]) - 1;
        int hi = index_bin(f, mx[f]) + 1;
        if (lo < 0) lo = 0;
        if (hi >= kIndexBins) hi = kIndexBins - 1;
        if (hi < lo) std::swap(hi, lo);
        cells *= (hi - lo + 1);
        if (cells > kIndexMaxCells) return true;
    }
    return false;
}

void WorkloadProfiler::index_insert_row(int row) {
    if (row < 0 || row >= static_cast<int>(policy_matrix.size())) return;
    const PolicyRecord& rec = policy_matrix[static_cast<size_t>(row)];
    if (rec.winner_arm < 0) return;
    if (!rec.has_box) {
        policy_wide.push_back(row);
        return;
    }
    const auto mn = to_vector(rec.feat_min);
    const auto mx = to_vector(rec.feat_max);
    int lo[3], hi[3];
    int cells = 1;
    for (int k = 0; k < 3; ++k) {
        const size_t f = static_cast<size_t>(kIndexAxes[k]);
        if (!feature_matchable(f)) {
            lo[k] = hi[k] = 0;
            continue;
        }
        lo[k] = index_bin(f, mn[f]) - 1;
        hi[k] = index_bin(f, mx[f]) + 1;
        if (lo[k] < 0) lo[k] = 0;
        if (hi[k] >= kIndexBins) hi[k] = kIndexBins - 1;
        if (hi[k] < lo[k]) std::swap(hi[k], lo[k]);
        cells *= (hi[k] - lo[k] + 1);
        if (cells > kIndexMaxCells) {
            policy_wide.push_back(row);
            return;
        }
    }
    for (int a = lo[0]; a <= hi[0]; ++a) {
        for (int b = lo[1]; b <= hi[1]; ++b) {
            for (int c = lo[2]; c <= hi[2]; ++c) {
                policy_grid[index_pack(a, b, c)].push_back(row);
            }
        }
    }
}

void WorkloadProfiler::rebuild_policy_index() {
    policy_grid.clear();
    policy_wide.clear();
    for (size_t i = 0; i < policy_matrix.size(); ++i) {
        index_insert_row(static_cast<int>(i));
    }
}

std::vector<int> WorkloadProfiler::index_candidates_point(
        const WorkloadFeatures& p) const {
    std::unordered_set<int> seen;
    for (int r : policy_wide) seen.insert(r);
    const auto v = to_vector(p);
    int b[3];
    for (int k = 0; k < 3; ++k) {
        b[k] = index_bin(static_cast<size_t>(kIndexAxes[k]),
                         v[static_cast<size_t>(kIndexAxes[k])]);
    }
    for (int da = -1; da <= 1; ++da) {
        for (int db = -1; db <= 1; ++db) {
            for (int dc = -1; dc <= 1; ++dc) {
                const int a = b[0] + da, bb = b[1] + db, c = b[2] + dc;
                if (a < 0 || a >= kIndexBins || bb < 0 || bb >= kIndexBins ||
                    c < 0 || c >= kIndexBins) continue;
                auto it = policy_grid.find(index_pack(a, bb, c));
                if (it == policy_grid.end()) continue;
                for (int r : it->second) seen.insert(r);
            }
        }
    }
    return {seen.begin(), seen.end()};
}

std::vector<int> WorkloadProfiler::index_candidates_box(
        const WorkloadFeatures& box_min,
        const WorkloadFeatures& box_max) const {
    std::unordered_set<int> seen;
    for (int r : policy_wide) seen.insert(r);
    const auto mn = to_vector(box_min);
    const auto mx = to_vector(box_max);
    int lo[3], hi[3];
    for (int k = 0; k < 3; ++k) {
        const size_t f = static_cast<size_t>(kIndexAxes[k]);
        if (!feature_matchable(f)) {
            lo[k] = hi[k] = 0;
            continue;
        }
        lo[k] = index_bin(f, mn[f]) - 1;
        hi[k] = index_bin(f, mx[f]) + 1;
        if (lo[k] < 0) lo[k] = 0;
        if (hi[k] >= kIndexBins) hi[k] = kIndexBins - 1;
        if (hi[k] < lo[k]) std::swap(hi[k], lo[k]);
    }
    for (int a = lo[0]; a <= hi[0]; ++a) {
        for (int b = lo[1]; b <= hi[1]; ++b) {
            for (int c = lo[2]; c <= hi[2]; ++c) {
                auto it = policy_grid.find(index_pack(a, b, c));
                if (it == policy_grid.end()) continue;
                for (int r : it->second) seen.insert(r);
            }
        }
    }
    return {seen.begin(), seen.end()};
}

void WorkloadProfiler::clamp_thresholds() {
    if (!std::isfinite(shift_threshold)) shift_threshold = MIN_SHIFT_THRESHOLD;
    if (!std::isfinite(matrix_epsilon)) matrix_epsilon = MIN_MATRIX_EPSILON;
    shift_threshold = std::min(MAX_SHIFT_THRESHOLD,
                               std::max(MIN_SHIFT_THRESHOLD, shift_threshold));
    matrix_epsilon = std::min(MAX_MATRIX_EPSILON,
                              std::max(MIN_MATRIX_EPSILON, matrix_epsilon));
}

void WorkloadProfiler::update_online_threshold(double current_delta) {
    // TRAIN and TEST use the same unsupervised clock. Freezing the final
    // TRAIN threshold made mask-dependent values such as 0.04 and 1.0 split
    // the same TEST trace into 6,000 versus 600 regimes. Adapting here reads
    // feature motion only—never cache misses, shadow outcomes, or rewards—so
    // it aligns segmentation without fitting to TEST performance.
    double diff = current_delta - ema_delta_mean;
    ema_delta_mean += alpha * diff;
    ema_delta_var = (1.0 - alpha) * (ema_delta_var + alpha * diff * diff);
    
    double std_dev = std::sqrt(std::max(0.0, ema_delta_var));
    shift_threshold = ema_delta_mean + (shift_sigma * std_dev);
    matrix_epsilon = std::max(0.01, ema_delta_mean * 0.5);
    // A drifting ruler used to push these into the hundreds, so feature drift
    // never fired and almost nothing reached an upsert.
    clamp_thresholds();
}

void WorkloadProfiler::update_feature_statistics(const WorkloadFeatures& f) {
    // Held-out TEST must keep the training distance ruler fixed.
    if (current_mode == ProfilerMode::TEST) return;
    auto vec = to_vector(f);
    for (size_t i : active_idx) {
        double diff = vec[i] - feature_means[i];
        feature_means[i] += alpha * diff;
        double var = (1.0 - alpha) * (feature_vars[i] + alpha * diff * diff);
        if (!std::isfinite(var)) var = FEATURE_VAR_FLOOR;
        feature_vars[i] = std::max(var, FEATURE_VAR_FLOOR);
    }
}

double WorkloadProfiler::calculate_aggregate_distance(const WorkloadFeatures& f1,
                                                      const WorkloadFeatures& f2) {
    auto v1 = to_vector(f1);
    auto v2 = to_vector(f2);
    double weighted_sum = 0.0;
    double total_weight = 0.0;
    
    for (size_t i : active_idx) {
        if (!feature_usable(i)) continue;
        double norm_diff = std::min(MAX_NORM_DIFF,
                                    std::abs(v1[i] - v2[i]) / feature_std(i));
        
        weighted_sum += feature_importance_weights[i] * norm_diff;
        total_weight += feature_importance_weights[i];
    }
    return total_weight > 0.0 ? weighted_sum / total_weight : 1e300;
}

void WorkloadProfiler::save_config() {
    if (config_filename.empty()) return;
    if (!ensure_parent_dir(config_filename)) return;
    std::string tmp = config_filename + ".tmp";
    std::ofstream file(tmp, std::ios::trunc);
    if (!file.is_open()) return;
    // v3: thresholds + the robust ruler (median and 1-sigma spread) used by
    // distance / match bands. v2 stored an EWMA variance instead.
    file << "v3\n";
    file << shift_threshold << "\n";
    file << matrix_epsilon << "\n";
    for (int i = 0; i < 15; ++i) {
        if (i) file << " ";
        file << feature_means[i];
    }
    file << "\n";
    for (int i = 0; i < 15; ++i) {
        if (i) file << " ";
        file << feature_scale[i];
    }
    file << "\n";
    file.close();

    int ret = std::rename(tmp.c_str(), config_filename.c_str());
    if (ret != 0) {
        std::ifstream src(tmp, std::ios::binary);
        std::ofstream dst(config_filename, std::ios::binary | std::ios::trunc);
        if (src.is_open() && dst.is_open()) {
            dst << src.rdbuf();
            dst.close();
        }
        src.close();
        std::remove(tmp.c_str());
    }
}

void WorkloadProfiler::load_config() {
    feature_scale_loaded = false;
    std::ifstream file(config_filename);
    if (!file.is_open()) return;

    std::string first;
    if (!(file >> first)) return;

    // v3 stores the robust sigma directly; v2 stored an EWMA variance whose
    // square root is the closest equivalent. Old format started with a float.
    if (first == "v2" || first == "v3") {
        const bool v3 = (first == "v3");
        if (!(file >> shift_threshold)) return;
        if (!(file >> matrix_epsilon)) return;
        for (int i = 0; i < 15; ++i) {
            if (!(file >> feature_means[i])) return;
        }
        for (int i = 0; i < 15; ++i) {
            double x = 0.0;
            if (!(file >> x)) return;
            if (!std::isfinite(x) || x < 0.0) x = 0.0;
            // A v2 variance pinned at the floor means the old estimator gave
            // up on the feature; carry that through as "no spread" rather
            // than inventing a sigma of 0.01.
            if (!v3) x = (x <= FEATURE_VAR_FLOOR) ? 0.0 : std::sqrt(x);
            feature_scale[i] = x;
            loaded_scale[i] = x;
            feature_vars[i] = (x > 0.0) ? x * x : 1.0;
        }
        feature_scale_loaded = true;
        clamp_thresholds();
    } else {
        // Backward compatible: two floats only.
        try {
            shift_threshold = std::stod(first);
        } catch (...) {
            return;
        }
        if (!(file >> matrix_epsilon)) return;
        // Optional trailing means/vars (partial upgrade without v2 tag).
        double probe = 0.0;
        if (file >> probe) {
            feature_means[0] = probe;
            bool ok = true;
            for (int i = 1; i < 15 && ok; ++i) ok = static_cast<bool>(file >> feature_means[i]);
            for (int i = 0; i < 15 && ok; ++i) {
                double x = 0.0;
                ok = static_cast<bool>(file >> x);
                if (!ok) break;
                if (!std::isfinite(x) || x <= FEATURE_VAR_FLOOR) x = 0.0;
                else x = std::sqrt(x);
                feature_scale[i] = x;
                loaded_scale[i] = x;
                feature_vars[i] = (x > 0.0) ? x * x : 1.0;
            }
            if (ok) feature_scale_loaded = true;
        }
        clamp_thresholds();
    }
    file.close();
}

void WorkloadProfiler::write_features_csv(std::ostream& file,
                                          const WorkloadFeatures& f) const {
    file << f.write_ratio << "," << f.avg_request_size << "," << f.size_variance << ","
         << f.singleton_ratio << "," << f.popularity_skewness << "," << f.avg_frequency << ","
         << f.object_diversity << "," << f.avg_reuse_distance << "," << f.sequentiality_ratio << ","
         << f.out_cache_hit_rate << "," << f.request_rate << "," << f.burstiness_index << ","
         << f.arrival_time_variance << "," << f.working_set_byte_delta << "," << f.scan_ratio;
}

void WorkloadProfiler::write_active_csv(std::ostream& file,
                                        const WorkloadFeatures& f) const {
    const auto v = to_vector(f);
    bool first = true;
    for (size_t i : active_idx) {
        if (!first) file << ",";
        first = false;
        file << v[i];
    }
}

void WorkloadProfiler::save_policy_matrix() {
    if (policy_filename.empty()) return;

    if (!ensure_parent_dir(policy_filename)) return;

    std::string tmp = policy_filename + ".tmp";
    std::ofstream file(tmp, std::ios::trunc);
    if (!file.is_open()) return;

    // v8 stores the box over the mask's features only. Row creation is
    // geography-only; reward updates the policy attached to an existing row.
    // The other coordinates are never read by distance or containment.
    file << "knn_v8,mask=" << feature_mask_str
         << ",box_min_active,box_max_active,winner_arm,best_miss_rate,"
            "best_delta,queue,queue_next,tried_arms_mask\n";
    for (const auto& record : policy_matrix) {
        file << "knn_v8,box,";
        write_active_csv(file, record.feat_min);
        file << ",";
        write_active_csv(file, record.feat_max);
        file << "," << record.winner_arm
             << "," << record.best_miss_rate
             << "," << record.best_delta;
        std::vector<uint8_t> queue = record.arm_queue;
        const int k = std::max(1, static_cast<int>(current_mab_k));
        if (static_cast<int>(queue.size()) != k) {
            queue.resize(static_cast<size_t>(k));
            for (int i = 0; i < k; ++i) queue[static_cast<size_t>(i)] = static_cast<uint8_t>(i);
        }
        for (uint8_t arm : queue) {
            file << "," << static_cast<int>(arm);
        }
        file << "," << static_cast<int>(record.queue_next)
             << "," << record.tried_arms_mask << "\n";
    }
    file.close();

    int ret = std::rename(tmp.c_str(), policy_filename.c_str());
    if (ret != 0) {
        std::ifstream src(tmp, std::ios::binary);
        std::ofstream dst(policy_filename, std::ios::binary | std::ios::trunc);
        if (src.is_open() && dst.is_open()) {
            dst << src.rdbuf();
            dst.close();
        }
        src.close();
        std::remove(tmp.c_str());
    }
}

void WorkloadProfiler::load_policy_matrix() {
    policy_matrix.clear();
    policy_grid.clear();
    policy_wide.clear();
    std::ifstream file(policy_filename);
    if (!file.is_open()) return;

    const std::string row_tag = "knn_v8,box,";
    const std::string want_hdr = "knn_v8,mask=" + feature_mask_str + ",";
    const size_t nf = active_idx.size();

    std::string line;
    size_t rejected_old_rows = 0;
    bool mask_mismatch = false;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (line.rfind("knn_v8,mask=", 0) == 0) {
            // A table built under a different mask has a different column
            // meaning per row; loading it would silently compare unrelated
            // coordinates. Drop it and start fresh.
            if (line.rfind(want_hdr, 0) != 0) {
                mask_mismatch = true;
                break;
            }
            continue;
        }
        if (line.rfind(row_tag, 0) != 0) {
            ++rejected_old_rows;
            continue;
        }

        PolicyRecord record;
        std::vector<double> all_nums;
        std::stringstream ss(line.substr(row_tag.size()));
        std::string val;
        bool bad = false;
        while (std::getline(ss, val, ',')) {
            try {
            all_nums.push_back(std::stod(val));
            } catch (...) {
                bad = true;
                break;
            }
        }
        if (bad) continue;

        const int k = std::max(1, static_cast<int>(current_mab_k));
        // nf min + nf max + winner + miss + delta + queue[k] + next + mask.
        const size_t base = 2 * nf;
        if (all_nums.size() != base + 5 + static_cast<size_t>(k)) continue;
        std::vector<double> lo(15, 0.0), hi(15, 0.0), mid(15, 0.0);
        for (size_t j = 0; j < nf; ++j) {
            const size_t i = active_idx[j];
            lo[i] = all_nums[j];
            hi[i] = all_nums[nf + j];
            mid[i] = 0.5 * (lo[i] + hi[i]);
        }
        from_vector(lo, record.feat_min);
        from_vector(hi, record.feat_max);
        from_vector(mid, record.features);
        record.has_box = true;
        record.winner_arm = static_cast<int>(all_nums[base]);
        record.best_miss_rate = all_nums[base + 1];
        record.best_delta = all_nums[base + 2];
        bool queue_ok = record.winner_arm >= -1 &&
                        record.winner_arm < k;
        std::vector<bool> seen(static_cast<size_t>(k), false);
        record.arm_queue.assign(static_cast<size_t>(k), 0);
        for (int i = 0; i < k; ++i) {
            const int arm = static_cast<int>(all_nums[base + 3 + static_cast<size_t>(i)]);
            if (arm < 0 || arm >= k || seen[static_cast<size_t>(arm)]) {
                queue_ok = false;
                break;
            }
            seen[static_cast<size_t>(arm)] = true;
            record.arm_queue[static_cast<size_t>(i)] = static_cast<uint8_t>(arm);
        }
        record.queue_next =
            static_cast<uint8_t>(all_nums[base + 3 + static_cast<size_t>(k)]);
        record.tried_arms_mask =
            static_cast<uint16_t>(all_nums[base + 4 + static_cast<size_t>(k)]);
        const uint16_t valid_mask =
            static_cast<uint16_t>((1u << static_cast<unsigned>(k)) - 1u);
        if (!queue_ok || record.queue_next >= static_cast<uint8_t>(k)) continue;
        if ((record.tried_arms_mask & ~valid_mask) != 0 ||
            record.tried_arms_mask == 0) continue;

        policy_matrix.push_back(record);
        // If the train scale was restored from config, do not reseeding it from
        // row centers (would rewrite σ and break TEST distance bands).
        if (!feature_scale_loaded) {
        update_feature_statistics(record.features);
        }
    }
    file.close();
    if (mask_mismatch) {
        policy_matrix.clear();
        std::cerr << "KNN policy feature-mask mismatch in " << policy_filename
                  << "; starting fresh for mask " << feature_mask_str
                  << std::endl;
    } else if (rejected_old_rows > 0) {
        std::cerr << "KNN policy schema mismatch: ignored "
                  << rejected_old_rows
                  << " old row(s); start fresh with knn_v8" << std::endl;
    }
    compact_policy_matrix();
    rebuild_policy_index();
}

std::vector<double> WorkloadProfiler::find_closest_policy(
        const WorkloadFeatures& current, bool* out_in_box,
        int* out_policy_row) {
    if (out_in_box) *out_in_box = false;
    if (out_policy_row) *out_policy_row = -1;
    if (policy_matrix.empty()) {
        last_match_dist = 1e300;
        return std::vector<double>();
    }

    struct Candidate {
        int idx;
        double dist;
        bool contained;
    };
    std::vector<Candidate> candidates;
    auto add_cand = [&](int i) {
        if (i < 0 || i >= static_cast<int>(policy_matrix.size())) return;
        if (policy_matrix[static_cast<size_t>(i)].winner_arm < 0) return;
        candidates.push_back({
            i,
            calculate_aggregate_distance(current, policy_matrix[static_cast<size_t>(i)].features),
            record_contains(current, policy_matrix[static_cast<size_t>(i)])
        });
    };
    if (index_linear_faster(IndexUse::Lookup)) {
        for (size_t i = 0; i < policy_matrix.size(); ++i) add_cand(static_cast<int>(i));
    } else {
        for (int i : index_candidates_point(current)) add_cand(i);
    }
    if (candidates.empty()) {
        last_match_dist = 1e300;
        return std::vector<double>();
    }
    // Only the nearest five contribute. Containment remains an audit signal,
    // not a ranking override: a broad box must not outrank a much closer
    // regime merely because the current point lies inside its padded bounds.
    const auto by_rank = [](const Candidate& a, const Candidate& b) {
        if (std::abs(a.dist - b.dist) > 1e-12) return a.dist < b.dist;
        if (a.contained != b.contained) return a.contained > b.contained;
        return a.idx < b.idx;
    };
    const size_t keep = std::min<size_t>(5, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + keep,
                      candidates.end(), by_rank);
    const bool in_box = candidates.front().contained;
    const int best_idx = candidates.front().idx;
    const double best_dist = candidates.front().dist;
    if (out_in_box) *out_in_box = in_box;
    if (out_policy_row) *out_policy_row = best_idx;
    last_match_dist = best_dist;
    active_nn_dist.fill(-1.0);
    std::vector<double> blend(current_mab_k, 0.0);
    const size_t count = std::min<size_t>(5, candidates.size());
    for (size_t i = 0; i < count; ++i) {
        const PolicyRecord& rec =
            policy_matrix[static_cast<size_t>(candidates[i].idx)];
        active_nn_dist[i] = candidates[i].dist;
        // Distance only. The 0.05 offset keeps a zero-distance row finite.
        // best_delta already chose this row's stored arm; it does not vote.
        const double d = 0.05 + std::max(0.0, candidates[i].dist);
        const double weight = 1.0 / (d * d);
        if (rec.winner_arm < 0 ||
            rec.winner_arm >= static_cast<int>(blend.size())) continue;
        blend[static_cast<size_t>(rec.winner_arm)] += weight;
    }

    double total = 0.0;
    for (double value : blend) total += value;
    if (total <= 0.0) {
        // No usable neighbour arm. Replay the closest stored arm.
        blend.assign(current_mab_k, 0.0);
        const int fallback =
            policy_matrix[static_cast<size_t>(best_idx)].winner_arm;
        blend[static_cast<size_t>(fallback)] = 1.0;
    } else {
        for (double& value : blend) value /= total;
    }
    return blend;
}

uint8_t WorkloadProfiler::select_regime_arm(bool* out_matched) {
    if (out_matched) *out_matched = false;

    if (current_mode == ProfilerMode::TRAIN) {
        if (!is_profile_ready()) {
            active_train_policy_row = -1;
            active_train_arm = 3;
            return 3;
        }

        int row_idx = -1;
        double best_dist = 1e300;
        auto consider = [&](int i) {
            if (i < 0 || i >= static_cast<int>(policy_matrix.size())) return;
            if (!record_contains(current_features, policy_matrix[static_cast<size_t>(i)])) return;
            const double dist = calculate_aggregate_distance(
                current_features, policy_matrix[static_cast<size_t>(i)].features);
            if (dist < best_dist) {
                best_dist = dist;
                row_idx = i;
            }
        };
        if (index_linear_faster(IndexUse::Lookup)) {
            for (size_t i = 0; i < policy_matrix.size(); ++i) consider(static_cast<int>(i));
        } else {
            for (int i : index_candidates_point(current_features)) consider(i);
        }
        // No row covers this regime yet. Do not seed one here: a point box
        // almost never contains the next regime, so seeding at open would add
        // a row per regime. The close will store the box it actually measured.
        if (row_idx < 0) {
            const uint8_t arm = static_cast<uint8_t>(
                train_arm_rotation % std::max<uint8_t>(1, current_mab_k));
            train_arm_rotation = static_cast<uint8_t>(
                (train_arm_rotation + 1) % std::max<uint8_t>(1, current_mab_k));
            active_train_policy_row = -1;
            active_train_arm = static_cast<int>(arm);
            return arm;
        }

        PolicyRecord& rec = policy_matrix[static_cast<size_t>(row_idx)];
        if (rec.arm_queue.size() != static_cast<size_t>(current_mab_k)) {
            initialize_arm_queue(rec);
        }
        const int n = static_cast<int>(rec.arm_queue.size());
        if (n <= 0) {
            active_train_policy_row = -1;
            active_train_arm = 3;
            return 3;
        }
        if (rec.queue_next >= static_cast<uint8_t>(n)) rec.queue_next = 0;
        const uint8_t arm = rec.arm_queue[rec.queue_next];
        rec.queue_next = static_cast<uint8_t>(
            (static_cast<int>(rec.queue_next) + 1) % n);
        active_train_policy_row = row_idx;
        active_train_arm = arm;
        // The table is saved when the trace ends. run_msr_feature_screen.sh
        // resumes at trace granularity, so saving on every regime open would
        // rewrite the whole table thousands of times and buy nothing.
        if (out_matched) *out_matched = rec.winner_arm >= 0;
        return arm;
    }

    if (!is_profile_ready()) {
        active_policy_match = false;
        active_match_refused = false;
        active_match_dist = 1e300;
        active_match_band = N_MATCH_BANDS - 1;
        active_nn_dist.fill(-1.0);
        active_policy_row = -1;
        active_policy_arm = 3;
        active_policy_type = classify_regime_type(current_features);
        active_policy_weights.assign(current_mab_k, 0.01);
        active_policy_weights[3] = 1.0;
        phase_score_active = true;
        score_requests = score_bytes = score_miss_bytes =
            score_shadow_miss_bytes = 0;
        return 3;
    }

    int policy_row = -1;
    std::vector<double> matched =
        find_closest_policy(current_features, nullptr, &policy_row);
    int winner = 3; // Empty table/no completed TRAIN row: Explore.
    const bool usable = !matched.empty();
    if (usable) {
        winner = 0;
        for (int arm = 1; arm < static_cast<int>(matched.size()); ++arm) {
            if (matched[static_cast<size_t>(arm)] >
                matched[static_cast<size_t>(winner)]) winner = arm;
        }
    }

    active_policy_match = usable;
    // There is no hard distance refusal: every completed KNN row can vote,
    // with distant rows receiving less weight.
    active_match_refused = false;
    active_match_dist = matched.empty() ? 1e300 : last_match_dist;
    if (matched.empty()) active_nn_dist.fill(-1.0);
    active_match_band = match_band_index(!matched.empty(), active_match_dist);
    active_policy_row = policy_row;
    active_policy_arm = winner;
    active_policy_type =
        (policy_row >= 0 && policy_row < static_cast<int>(policy_matrix.size()))
        ? classify_regime_type(policy_matrix[static_cast<size_t>(policy_row)].features)
        : N_REGIME_TYPES - 1;
    active_policy_weights = matched;
    active_inject_features = current_features;
    phase_score_active = true;
    score_requests = score_bytes = score_miss_bytes = score_shadow_miss_bytes = 0;
    if (out_matched) *out_matched = usable;
    return static_cast<uint8_t>(winner);
}

bool WorkloadProfiler::add_request(uint64_t id, uint32_t size, bool is_write, double timestamp, 
                                   bool out_cache_hit, bool is_miss, bool shadow_is_miss,
                                   const std::vector<double>& current_weights,
                                   bool arm_committed,
                                   std::vector<double>& out_new_weights,
                                   bool* out_regime_reset) {
    (void)out_new_weights;
    if (out_regime_reset) *out_regime_reset = false;
    last_regime_close_reason.clear();
    
    double dt = (local_seq > 0) ? (timestamp - last_timestamp) : 0.0;
    // Block traces address by offset, so a sequential access advances by the
    // previous request's length, not by one. Dense integer keys still match
    // because any size >= 1 admits last_id + 1.
    bool is_seq = (local_seq > 0 && id > last_id &&
                   id <= last_id + static_cast<uint64_t>(last_size));
    // last_seen_seq is a per-object map touched on every request and exists
    // only to feed avg_reuse_distance. When the mask drops that feature the
    // whole map goes with it.
    double r_dist = 0.0;
    if (need_reuse_distance) {
        auto it = last_seen_seq.find(id);
        if (it != last_seen_seq.end()) {
            r_dist = static_cast<double>(local_seq - it->second);
        }
    }
    bool first_time = (id_counts[id] == 0);

    if (is_write) write_count++;
    size_sum += size;
    size_sq_sum += static_cast<double>(size) * size;
    if (out_cache_hit) out_cache_hits++;
    if (is_seq) sequential_count++;
    if (r_dist > 0.0) { total_reuse_dist += r_dist; reuse_count++; }
    if (need_time_features && dt > 0.0) {
        dt_sum += dt; dt_sq_sum += dt * dt; dt_count++;
    }
    if (first_time) { 
        first_time_seen_count++; 
        if (need_working_set) {
        unique_id_sizes[id] = size;
        working_set_bytes += size; 
        }
    }

    size_t current_f = id_counts[id];
    if (current_f > 0) freq_sq_sum -= static_cast<double>(current_f) * current_f;
    if (current_f == 1) singleton_count--;
    
    size_t new_f = current_f + 1;
    freq_sq_sum += static_cast<double>(new_f) * new_f;
    if (new_f == 1) singleton_count++;

    id_counts[id]++;
    if (need_reuse_distance) last_seen_seq[id] = local_seq;
    window.push_back({id, size, is_write, timestamp, out_cache_hit, r_dist, is_seq, dt, first_time});
    
    last_id = id;
    last_size = size;
    last_timestamp = timestamp;
    local_seq++;

    if (window.size() > window_size) {
        RequestLog old = window.front();
        window.pop_front();

        if (old.is_write) write_count--;
        size_sum -= old.size;
        size_sq_sum -= static_cast<double>(old.size) * old.size;
        if (old.out_cache_hit) out_cache_hits--;
        if (old.is_sequential) sequential_count--;
        if (old.reuse_dist > 0.0) { total_reuse_dist -= old.reuse_dist; reuse_count--; }
        if (need_time_features && old.dt > 0.0) {
            dt_sum -= old.dt; dt_sq_sum -= old.dt * old.dt; dt_count--;
        }
        if (old.first_time) first_time_seen_count--;

        size_t old_f = id_counts[old.id];
        freq_sq_sum -= static_cast<double>(old_f) * old_f;
        if (old_f == 1) singleton_count--;

        size_t old_new_f = old_f - 1;
        if (old_new_f > 0) freq_sq_sum += static_cast<double>(old_new_f) * old_new_f;
        if (old_new_f == 1) singleton_count++;

        id_counts[old.id]--;
        if (id_counts[old.id] == 0) {
            id_counts.erase(old.id);
            if (need_working_set) {
                auto us = unique_id_sizes.find(old.id);
                if (us != unique_id_sizes.end()) {
                    working_set_bytes -= us->second;
                    unique_id_sizes.erase(us);
                }
            }
        }
    }

    // Byte shock is now counted, not acted on. One outsized request used to
    // be able to close a regime on its own, which is half of why the median
    // regime length was a single request.
    const bool byte_shock = (shock_ref_size > 0.0 &&
                             static_cast<double>(size) >
                                 SHOCK_SIZE_MULT * shock_ref_size);
    if (byte_shock) phase_shock_count++;

    // The decision clock. Everything above this point is O(1) per request;
    // everything below runs once per REGIME_TICK requests.
    const bool tick = (local_seq % REGIME_TICK) == 0;
    if (!tick) {
        request_counter++;
        phase_requests++;
        phase_bytes += size;
        if (is_miss) { phase_misses++; phase_miss_bytes += size; }
        if (shadow_is_miss) {
            phase_shadow_misses++;
            phase_shadow_miss_bytes += size;
        }
        // The score window and the stable weights must follow the requests,
        // not the clock: the stored policy is whatever the arm was running
        // when the regime ended, and the delta must cover every committed
        // request rather than only those that landed on a tick.
        if (current_mode == ProfilerMode::TRAIN && arm_committed &&
            !phase_score_active) {
            phase_score_active = true;
            score_requests = 0;
            score_bytes = 0;
            score_miss_bytes = 0;
            score_shadow_miss_bytes = 0;
        }
        if (phase_score_active) {
            score_requests++;
            score_bytes += size;
            if (is_miss) score_miss_bytes += size;
            if (shadow_is_miss) score_shadow_miss_bytes += size;
        }
        if (arm_committed) {
            last_stable_features = current_features;
            last_stable_weights = current_weights;
            normalize_weights(last_stable_weights);
        }
        if (current_mode == ProfilerMode::TRAIN) {
            train_total_reqs++;
            if (is_miss) train_total_misses++;
        }
        if (phase_requests >= MIN_REGIME_LEN) phase_box_frozen = true;
        return false;
    }

    tick_counter++;
    recalculate_features();
    if (!scale_ready && !feature_scale_loaded) {
    update_feature_statistics(current_features);
    }
    if (current_mode == ProfilerMode::TRAIN) {
        scale_reservoir_add(to_vector(current_features));
        ++ticks_since_scale;
        // Take the ruler as soon as there are enough samples to trust it,
        // then re-derive it rarely: the sort is the only super-linear work
        // left on this path.
        const bool first_ruler =
            (!scale_ready && scale_reservoir.size() >= SCALE_MIN_SAMPLES);
        if (first_ruler || ticks_since_scale >= SCALE_REFRESH_TICKS) {
            ticks_since_scale = 0;
            refresh_feature_scale();
        }
    }

    if (tick_counter > 1) {
        double delta = calculate_aggregate_distance(current_features, prev_window_features);
        update_online_threshold(delta);
        prev_window_features = current_features;
    }

    request_counter++;
    if (current_mode == ProfilerMode::TRAIN &&
        flush_interval > 0 && request_counter > 0 &&
        (request_counter % flush_interval == 0)) {
        save_policy_matrix();
        save_config();
    }

    if (!phase_box_init) {
        open_new_regime();
    }
    bool warmup_reset = false;
    const bool profile_ready = window.size() >= window_size;
    if (profile_ready && !profile_ready_seen) {
        profile_ready_seen = true;
        if (phase_requests > 0) {
            last_regime_close_reason = "profile_warmup";
            close_current_regime("profile_warmup", nullptr);
            open_new_regime();
            warmup_reset = true;
            if (out_regime_reset) *out_regime_reset = true;
        }
    }
    phase_ticks++;
    double current_shift = calculate_aggregate_distance(current_features, phase_start_features);

    // The area is the min/max collected over the first MIN_REGIME_LEN
    // requests. After that, a tick outside that range is a new regime.
    // Two such ticks are required so one noisy sample does not cut it.
    const bool shock_tick =
        (phase_shock_count >
         static_cast<uint64_t>(SHOCK_TICK_SHARE * REGIME_TICK));
    const bool outside_box = phase_box_frozen &&
        !point_in_box(current_features, phase_box_min, phase_box_max);
    if (outside_box || shock_tick) {
        drift_ticks++;
    } else {
        drift_ticks = 0;
    }
    phase_shock_count = 0;

    const bool drift_confirmed =
        (drift_ticks >= DRIFT_CONFIRM && phase_requests >= MIN_REGIME_LEN);
    // A perfectly stable trace would otherwise never produce a second
    // training sample.
    const bool too_long = (phase_requests >= MAX_REGIME_LEN);
    bool feature_regime_reset =
        (!warmup_reset && phase_requests > 0 && (drift_confirmed || too_long));
    if (feature_regime_reset) drift_ticks = 0;
    const char* drift_action = nullptr;
    double drift_delta = 0.0;
    std::vector<double> closed_weights;
    if (feature_regime_reset) {
        const char* why = too_long ? "max_len" : "feature_drift";
        last_regime_close_reason = why;
        closed_weights = last_stable_weights;
        drift_action = close_current_regime(why, &drift_delta);
        open_new_regime();
        if (out_regime_reset) *out_regime_reset = true;
    }
    // The first minimum-length window defines the area. Requests after that
    // stay on the same regime until a feature leaves that saved range.
    if (!phase_box_frozen) {
        expand_box(phase_box_min, phase_box_max, current_features);
    }

    phase_requests++;
    if (phase_requests >= MIN_REGIME_LEN) phase_box_frozen = true;
    phase_bytes += size;
    if (is_miss) {
        phase_misses++;
        phase_miss_bytes += size;
    }
    if (shadow_is_miss) {
        phase_shadow_misses++;
        phase_shadow_miss_bytes += size;
    }
    // First committed request in this regime: start the store-score window
    // and drop any prior investigate traffic from the upsert delta.
    if (current_mode == ProfilerMode::TRAIN && arm_committed &&
        !feature_regime_reset && !phase_score_active) {
        phase_score_active = true;
        score_requests = 0;
        score_bytes = 0;
        score_miss_bytes = 0;
        score_shadow_miss_bytes = 0;
    }
    if (phase_score_active && !feature_regime_reset) {
        score_requests++;
        score_bytes += size;
        if (is_miss) score_miss_bytes += size;
        if (shadow_is_miss) score_shadow_miss_bytes += size;
    }
    if (arm_committed && !feature_regime_reset) {
            last_stable_features = current_features;
            last_stable_weights = current_weights;
        normalize_weights(last_stable_weights);
    }

    if (current_mode == ProfilerMode::TRAIN) {
        train_total_reqs++;
        if (is_miss) {
            train_total_misses++;
        }

        double weight_delta = 0.0;
        if (!last_stable_weights.empty() && last_stable_weights.size() == current_weights.size()) {
            for (size_t i = 0; i < current_weights.size(); ++i) {
                weight_delta = std::max(weight_delta,
                                        std::abs(current_weights[i] - last_stable_weights[i]));
            }
        }

        // Feature clock only. Arm weight moves do NOT close the regime, and
        // byte shock is folded into the tick drift vote above rather than
        // cutting the regime here.
        if (feature_regime_reset && train_log.enabled()) {
            train_log.log_learning(local_seq, drift_action,
                                   policy_matrix.size(), drift_delta,
                                   0.0, -1, closed_weights);
        }
        // Policy is saved at trace end. Rewriting the whole table on every
        // stored close is what made a growing file stall the run on /mnt/c.

        if (train_log.enabled() && train_log.interval() > 0 &&
            (local_seq % train_log.interval()) == 0) {
            double phase_miss = (phase_requests > 0)
                ? (static_cast<double>(phase_misses) / phase_requests) : 0.0;
            double cum_miss_now = (train_total_reqs > 0)
                ? (static_cast<double>(train_total_misses) / train_total_reqs) : 0.0;
            train_log.log_progress(local_seq, policy_matrix.size(), phase_miss, cum_miss_now,
                                   current_shift, shift_threshold, weight_delta,
                                   current_weights);
        }
        return false;
            } else {
        // TEST selection happens once at regime open through
        // select_regime_arm(); never probe or reselect inside this regime.
        return false;
    }
}

void WorkloadProfiler::flush_to_disk() {
    if (current_mode == ProfilerMode::TRAIN) {
        flush_regime_events(true);
        rewrite_feature_stats();
    save_policy_matrix();
    save_config();
    } else {
        flush_test_match_events(true);
    }
    train_log.flush();
}

void WorkloadProfiler::recalculate_features() {
    if (window.empty()) return;
    size_t total = window.size();
    size_t unique = id_counts.size();
    if (unique == 0) return;

    // Byte-shock detection needs a size reference even when the size feature
    // itself is masked off, so this one average is always maintained.
    const double mean_sz = size_sum / total;
    shock_ref_size = mean_sz;

    // Only the mask's features are computed. An unmasked feature is never
    // read by distance, containment or the stored box, so filling it in was
    // pure cost: 15 fields per evaluation regardless of the experiment.
    if (feature_active[0]) {
        current_features.write_ratio = static_cast<double>(write_count) / total;
    }
    if (feature_active[1]) current_features.avg_request_size = mean_sz;
    if (feature_active[2]) {
    double var_sz = (size_sq_sum / total) - (mean_sz * mean_sz);
    current_features.size_variance = (var_sz > 0.0) ? var_sz : 0.0;
    }
    if (feature_active[3]) {
        current_features.singleton_ratio =
            static_cast<double>(singleton_count) / total;
    }
    const double mean_f = static_cast<double>(total) / unique;
    if (feature_active[5]) current_features.avg_frequency = mean_f;
    if (feature_active[4]) {
    double var_f = (freq_sq_sum / unique) - (mean_f * mean_f);
        current_features.popularity_skewness =
            (var_f > 0.0) ? std::sqrt(var_f) : 0.0;
    }
    if (feature_active[6]) {
        current_features.object_diversity =
            static_cast<double>(unique) / total;
    }
    if (feature_active[7]) {
        current_features.avg_reuse_distance =
            (reuse_count > 0) ? (total_reuse_dist / reuse_count) : 0.0;
    }
    if (feature_active[8]) {
        current_features.sequentiality_ratio =
            static_cast<double>(sequential_count) / total;
    }
    if (feature_active[9]) {
        current_features.out_cache_hit_rate =
            static_cast<double>(out_cache_hits) / total;
    }

    if (need_time_features && dt_count > 0) {
        double mean_dt = dt_sum / dt_count;
        double var_dt = (dt_sq_sum / dt_count) - (mean_dt * mean_dt);
        if (var_dt < 0.0) var_dt = 0.0;

        // Track the trace's own typical gap, then report every time feature as
        // a ratio against it. Raw values are unusable across datasets because
        // one trace counts seconds and another counts 100 ns ticks.
        if (mean_dt > 0.0) {
            dt_scale = (dt_scale > 0.0)
                ? dt_scale + DT_SCALE_ALPHA * (mean_dt - dt_scale)
                : mean_dt;
        }

        const double scale = (dt_scale > 0.0) ? dt_scale : mean_dt;
        current_features.request_rate =
            (mean_dt > 0.0 && scale > 0.0) ? (scale / mean_dt) : 0.0;
        current_features.arrival_time_variance =
            (scale > 0.0) ? (var_dt / (scale * scale)) : 0.0;

        const double std_dt = std::sqrt(var_dt);
        current_features.burstiness_index =
            (mean_dt > 0.0) ? (std_dt / mean_dt) : 0.0;
    } else if (need_time_features) {
        current_features.request_rate = 0.0;
        current_features.arrival_time_variance = 0.0;
        current_features.burstiness_index = 0.0;
    }

    if (feature_active[13]) {
    current_features.working_set_byte_delta = working_set_bytes;
    }
    if (feature_active[14]) {
        current_features.scan_ratio =
            static_cast<double>(first_time_seen_count) / total;
    }
}
