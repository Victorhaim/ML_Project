#include "TLCacheMAB.h"
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using namespace std;
using namespace TLCache;

TLCacheMABCache::~TLCacheMABCache() {
    const double mab_omr = total_req_count > 0
        ? static_cast<double>(total_miss_count) / total_req_count : 0.0;
    const double mab_bmr = total_bytes_req > 0
        ? static_cast<double>(total_bytes_miss) / total_bytes_req : 0.0;
    const double shadow_omr = total_req_count > 0
        ? static_cast<double>(total_shadow_misses) / total_req_count : 0.0;
    const double shadow_bmr = total_bytes_req > 0
        ? static_cast<double>(total_shadow_bytes_miss) / total_bytes_req : 0.0;
    // One aggregate line per trace, parsed by the experiment scripts.
    std::cout << "TLCACHE_MAB_SUMMARY"
              << " requests=" << total_req_count
              << " bytes=" << total_bytes_req
              << " mab_omr=" << mab_omr
              << " mab_bmr=" << mab_bmr
              << " shadow_omr=" << shadow_omr
              << " shadow_bmr=" << shadow_bmr
              << " arm_reselects=" << arm_reselect_count
              << " arm_slider_steps=" << arm_slider_steps
              << std::endl;
    if (diag.enabled()) {
        diag_maybe_snapshot("shutdown");
        diag.log_event(global_seq, "shutdown", "final_snapshot",
                       ema_miss_rate,
                       (total_req_count > 0)
                           ? (double)total_miss_count / (double)total_req_count
                           : 0.0,
                       mab_weights);
        diag.flush();
    }
}

const char* TLCacheMABCache::profiler_mode_cstr() const {
    return (profiler.get_mode() == ProfilerMode::TEST) ? "test" : "train";
}

// Retained only in source history for comparison with the superseded
// first-stage LightGBM experiment. It is not part of the KNN build.
#if 0
namespace {
std::string csv_quote(const std::string& value) {
    if (value.find_first_of(",\"\n") == std::string::npos) return value;
    std::string escaped = "\"";
    for (char c : value) escaped += (c == '"') ? "\"\"" : std::string(1, c);
    escaped += '"';
    return escaped;
}

void ensure_parent_directory(const std::string& filename) {
    const std::filesystem::path path(filename);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
}
}

std::vector<double> TLCacheMABCache::arm_model_row(uint8_t candidate_arm) const {
    std::vector<double> row = arm_regime_context;
    row.resize(ARM_CONTEXT_FEATURES, 0.0);
    for (int arm = 0; arm < 4; ++arm) {
        row.push_back(arm == candidate_arm ? 1.0 : 0.0);
    }
    return row;
}

double TLCacheMABCache::arm_model_predict(BoosterHandle model,
                                          const std::vector<double>& row) const {
    if (!model || static_cast<int>(row.size()) != ARM_MODEL_FEATURES) {
        throw std::runtime_error("invalid arm model or feature row");
    }
    int64_t out_len = 0;
    double prediction = 0.0;
    // 3L-Cache's LightGBM build has no start_iteration argument here.
    const int rc = LGBM_BoosterPredictForMat(
        model, row.data(), C_API_DTYPE_FLOAT64, 1, ARM_MODEL_FEATURES, 1,
        C_API_PREDICT_NORMAL, -1, "", &out_len, &prediction);
    if (rc != 0 || out_len != 1) {
        throw std::runtime_error("LightGBM arm prediction failed");
    }
    return prediction;
}

bool TLCacheMABCache::arm_model_load() {
    if (arm_mean_model_file.empty() || arm_downside_model_file.empty()) {
        std::cerr << "arm selector requires arm-mean-model and "
                     "arm-downside-model" << std::endl;
        return false;
    }
    int mean_iterations = 0;
    int downside_iterations = 0;
    if (LGBM_BoosterCreateFromModelfile(
            arm_mean_model_file.c_str(), &mean_iterations,
            &arm_mean_booster) != 0 ||
        LGBM_BoosterCreateFromModelfile(
            arm_downside_model_file.c_str(), &downside_iterations,
            &arm_downside_booster) != 0) {
        std::cerr << "unable to load arm selector model files" << std::endl;
        if (arm_mean_booster) {
            LGBM_BoosterFree(arm_mean_booster);
            arm_mean_booster = nullptr;
        }
        if (arm_downside_booster) {
            LGBM_BoosterFree(arm_downside_booster);
            arm_downside_booster = nullptr;
        }
        return false;
    }
    int mean_features = 0;
    int downside_features = 0;
    if (LGBM_BoosterGetNumFeature(arm_mean_booster, &mean_features) != 0 ||
        LGBM_BoosterGetNumFeature(arm_downside_booster,
                                  &downside_features) != 0 ||
        mean_features != ARM_MODEL_FEATURES ||
        downside_features != ARM_MODEL_FEATURES) {
        std::cerr << "arm selector feature mismatch: expected "
                  << ARM_MODEL_FEATURES << ", mean=" << mean_features
                  << ", downside=" << downside_features << std::endl;
        LGBM_BoosterFree(arm_mean_booster);
        LGBM_BoosterFree(arm_downside_booster);
        arm_mean_booster = nullptr;
        arm_downside_booster = nullptr;
        return false;
    }
    return true;
}

namespace {
std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quote = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_quote) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cur.push_back('"');
                    ++i;
                } else {
                    in_quote = false;
                }
            } else {
                cur.push_back(c);
            }
        } else if (c == '"') {
            in_quote = true;
        } else if (c == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

int csv_col(const std::vector<std::string>& header, const std::string& name) {
    for (int i = 0; i < static_cast<int>(header.size()); ++i) {
        if (header[static_cast<size_t>(i)] == name) return i;
    }
    return -1;
}

double clip_d(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

bool train_arm_booster(const std::vector<double>& x,
                       const std::vector<float>& y,
                       const std::vector<float>& w,
                       int nrow, int ncol,
                       std::unordered_map<std::string, std::string> params,
                       const std::string& out_path) {
    if (nrow <= 0 || ncol <= 0) return false;
    DatasetHandle dataset = nullptr;
    // Same convention as TLCache::train(): parameters go in as the map.
    if (LGBM_DatasetCreateFromMat(
            x.data(), C_API_DTYPE_FLOAT64, nrow, ncol, 1, params, nullptr,
            &dataset) != 0) {
        return false;
    }
    if (LGBM_DatasetSetField(dataset, "label", y.data(), nrow,
                             C_API_DTYPE_FLOAT32) != 0 ||
        LGBM_DatasetSetField(dataset, "weight", w.data(), nrow,
                             C_API_DTYPE_FLOAT32) != 0) {
        LGBM_DatasetFree(dataset);
        return false;
    }
    BoosterHandle booster = nullptr;
    if (LGBM_BoosterCreate(dataset, params, &booster) != 0) {
        LGBM_DatasetFree(dataset);
        return false;
    }
    const int rounds = std::max(1, std::stoi(params["num_iterations"]));
    for (int i = 0; i < rounds; ++i) {
        int finished = 0;
        if (LGBM_BoosterUpdateOneIter(booster, &finished) != 0) {
            LGBM_BoosterFree(booster);
            LGBM_DatasetFree(dataset);
            return false;
        }
        if (finished) break;
    }
    ensure_parent_directory(out_path);
    const int rc = LGBM_BoosterSaveModel(booster, 0, -1, out_path.c_str());
    LGBM_BoosterFree(booster);
    LGBM_DatasetFree(dataset);
    return rc == 0;
}
}

bool TLCacheMABCache::arm_selector_fit_from_csv() {
    if (arm_training_file.empty() ||
        !std::filesystem::exists(arm_training_file)) {
        std::cerr << "arm fit: training CSV missing: " << arm_training_file
                  << std::endl;
        return false;
    }
    if (arm_mean_model_file.empty() || arm_downside_model_file.empty()) {
        const std::filesystem::path parent =
            std::filesystem::path(arm_training_file).parent_path();
        if (arm_mean_model_file.empty()) {
            arm_mean_model_file = (parent / "mean_model.txt").string();
        }
        if (arm_downside_model_file.empty()) {
            arm_downside_model_file = (parent / "downside_model.txt").string();
        }
    }

    std::ifstream in(arm_training_file);
    if (!in) {
        std::cerr << "arm fit: cannot read " << arm_training_file << std::endl;
        return false;
    }
    std::string header_line;
    if (!std::getline(in, header_line)) {
        std::cerr << "arm fit: empty CSV" << std::endl;
        return false;
    }
    const std::vector<std::string> header = split_csv_line(header_line);
    const int bytes_col = csv_col(header, "bytes");
    const int delta_col = csv_col(header, "delta_bmr");
    const int conf_col = csv_col(header, "confidence");
    const int size_col = csv_col(header, "cache_size");
    std::array<int, ARM_MODEL_FEATURES> feat_col{};
    for (int i = 0; i < ARM_MODEL_FEATURES; ++i) {
        feat_col[static_cast<size_t>(i)] =
            csv_col(header, "f_" + std::to_string(i));
        if (feat_col[static_cast<size_t>(i)] < 0) {
            std::cerr << "arm fit: missing f_" << i << std::endl;
            return false;
        }
    }
    if (bytes_col < 0 || delta_col < 0 || conf_col < 0) {
        std::cerr << "arm fit: CSV missing bytes/delta_bmr/confidence"
                  << std::endl;
        return false;
    }

    std::vector<double> x;
    std::vector<float> y;
    std::vector<float> w;
    std::string line;
    int skipped = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const std::vector<std::string> cols = split_csv_line(line);
        if (static_cast<int>(cols.size()) <= feat_col.back()) {
            ++skipped;
            continue;
        }
        if (size_col >= 0 && !arm_cache_size_label.empty() &&
            cols[static_cast<size_t>(size_col)] != arm_cache_size_label) {
            continue;
        }
        double bytes = 0.0;
        double delta = 0.0;
        double conf = 0.0;
        try {
            bytes = std::stod(cols[static_cast<size_t>(bytes_col)]);
            delta = std::stod(cols[static_cast<size_t>(delta_col)]);
            conf = std::stod(cols[static_cast<size_t>(conf_col)]);
        } catch (...) {
            ++skipped;
            continue;
        }
        if (bytes <= 0.0) {
            ++skipped;
            continue;
        }
        for (int i = 0; i < ARM_MODEL_FEATURES; ++i) {
            x.push_back(std::stod(
                cols[static_cast<size_t>(feat_col[static_cast<size_t>(i)])]));
        }
        y.push_back(static_cast<float>(delta));
        w.push_back(static_cast<float>(
            clip_d(conf, 1e-6, 1.0) * clip_d(bytes, 1.0, 1e12)));
    }
    const int nrow = static_cast<int>(y.size());
    if (nrow <= 0) {
        std::cerr << "arm fit: no scored rows in " << arm_training_file
                  << std::endl;
        return false;
    }

    std::unordered_map<std::string, std::string> mean_params{
        {"boosting", "gbdt"},
        {"objective", "regression"},
        {"num_iterations", "200"},
        {"num_leaves", "31"},
        {"num_threads", "1"},
        {"learning_rate", "0.05"},
        {"min_data_in_leaf", "10"},
        {"verbosity", "-1"},
    };
    std::unordered_map<std::string, std::string> q_params = mean_params;
    q_params["objective"] = "quantile";
    q_params["alpha"] = "0.25";

    if (!train_arm_booster(x, y, w, nrow, ARM_MODEL_FEATURES, mean_params,
                           arm_mean_model_file) ||
        !train_arm_booster(x, y, w, nrow, ARM_MODEL_FEATURES, q_params,
                           arm_downside_model_file)) {
        std::cerr << "arm fit: LightGBM train/save failed" << std::endl;
        return false;
    }

    const std::filesystem::path meta_path =
        std::filesystem::path(arm_mean_model_file).parent_path() /
        "metadata.txt";
    std::ofstream meta(meta_path);
    if (!meta) {
        std::cerr << "arm fit: cannot write " << meta_path << std::endl;
        return false;
    }
    meta << "cache_size=" << arm_cache_size_label << "\n"
         << "feature_count=" << ARM_MODEL_FEATURES << "\n"
         << "n_rows=" << nrow << "\n"
         << "skipped=" << skipped << "\n"
         << "mean_model=" << arm_mean_model_file << "\n"
         << "downside_model=" << arm_downside_model_file << "\n";
    std::cout << "TLCACHE_MAB_FIT_DONE"
              << " rows=" << nrow
              << " skipped=" << skipped
              << " mean=" << arm_mean_model_file
              << " downside=" << arm_downside_model_file
              << std::endl;
    return true;
}

void TLCacheMABCache::arm_probe_build_schedule() {
    for (uint32_t cycle = 0; cycle < ARM_PROBE_CYCLES; ++cycle) {
        std::array<uint8_t, 4> order{{0, 1, 2, 3}};
        for (int i = 3; i > 0; --i) {
            const int j = std::min(
                i, static_cast<int>(mab_next_rand() * (i + 1)));
            std::swap(order[static_cast<size_t>(i)],
                      order[static_cast<size_t>(j)]);
        }
        for (int i = 0; i < 4; ++i) {
            arm_probe_schedule[cycle * 4 + static_cast<uint32_t>(i)] =
                order[static_cast<size_t>(i)];
        }
    }
}

void TLCacheMABCache::arm_build_context() {
    arm_regime_context = profiler.get_feature_vector();
    arm_regime_context.resize(15, 0.0);

    uint64_t resident_bytes = 0;
    for (const auto& meta : in_cache.metas) resident_bytes += meta._size;
    const double capacity = static_cast<double>(std::max<size_t>(1, getSize()));
    arm_regime_context.push_back(
        std::min(1.0, static_cast<double>(resident_bytes) / capacity));
    arm_regime_context.push_back(ema_miss_rate);
    arm_regime_context.push_back(miss_rate_slope);
    arm_regime_context.push_back(last_window_miss);
    arm_regime_context.push_back(
        previous_selector_arm < 0 ? -1.0 : previous_selector_arm / 3.0);
    arm_regime_context.push_back(previous_selector_delta);
}
#endif

bool TLCacheMABCache::arm_slider_active() const {
    return !mab_off && arm_strategy == ArmStrategy::MODE;
}

int TLCacheMABCache::arm_stream_cell_id() const {
    const WorkloadFeatures f = profiler.get_features();
    const int type = profiler.current_regime_type_idx();
    const double s = f.avg_request_size;
    int area = 7;
    if (s < 50.0) area = 0;
    else if (s < 100.0) area = 1;
    else if (s < 500.0) area = 2;
    else if (s < 1000.0) area = 3;
    else if (s < 2000.0) area = 4;
    else if (s < 10000.0) area = 5;
    else if (s < 50000.0) area = 6;
    return type * 8 + area;
}

uint8_t TLCacheMABCache::arm_rotate_start() {
    const int cell = arm_stream_cell_id();
    arm_cell_id = cell;
    uint8_t& next = arm_cell_next[cell];
    std::array<uint8_t, MODE_ARM_COUNT>& queue = arm_cell_queue[cell];
    if (next == 0) {
        for (uint8_t i = 0; i < MODE_ARM_COUNT; ++i) queue[i] = i;
        for (int i = static_cast<int>(MODE_ARM_COUNT) - 1; i > 0; --i) {
            const int j = std::min(
                i, static_cast<int>(mab_next_rand() * (i + 1)));
            std::swap(queue[static_cast<size_t>(i)],
                      queue[static_cast<size_t>(j)]);
        }
    }
    const uint8_t arm = queue[next];
    next = static_cast<uint8_t>((next + 1u) % MODE_ARM_COUNT);
    return arm;
}

void TLCacheMABCache::arm_set_arc_from_arm(uint8_t arm) {
    // {arc_lo, arc_width} as fractions of the ring from q.head, the LRU end.
    // An arc_lo of 0.75 with width 0.5 wraps past q.tail back to q.head, which
    // is how the two "ends" arms stay a single contiguous stretch.
    static const double arcs[MODE_ARM_COUNT][2] = {
        {0.750, 0.500}, // 0 Balanced ends  - across the ring junction
        {0.500, 0.500}, // 1 Tail+          - newest half
        {0.000, 0.500}, // 2 Head+          - oldest half
        {0.000, 1.000}, // 3 Explore        - whole ring
        {0.250, 0.500}, // 4 Lean middle    - middle half
        {0.625, 0.750}, // 5 Lean ends      - wider arc across the junction
    };
    int a = static_cast<int>(arm);
    if (a < 0 || a >= static_cast<int>(MODE_ARM_COUNT)) a = 3;
    arc_lo = arcs[a][0];
    arc_width = arcs[a][1];
    start_arc_lo = arc_lo;
    start_arc_width = arc_width;
    // A new arm means a new region, so restart the cursor and the lap.
    arc_pointer = UINT32_MAX;
    arc_lap_len = 0;
    arc_scan_pos = 0;
    arc_probe_used = 0;
    zone_samples.fill(0);
    zone_evictions.fill(0);
    start_arm = static_cast<uint8_t>(a);
    committed_arm = start_arm;
    last_mode_arm = start_arm;
    for (int i = 0; i < mab_k; ++i) {
        mab_weights[i] = (i == a) ? 1.0 : 1e-3;
    }
    mab_normalize_weights();
    mab_compute_probs();
    if (profiler.get_mode() == ProfilerMode::TEST) {
        const std::vector<double> arc{arc_lo, arc_width};
        profiler.set_window_start_audit(static_cast<int>(start_arm), arc);
    }
}

void TLCacheMABCache::arm_zone_begin_lap(uint32_t queue_len) {
    if (pos_zone_.size() < queue_len) {
        pos_zone_.resize(queue_len, 0);
        pos_zone_stamp_.resize(queue_len, 0);
    }
    if (++pos_zone_epoch_ == 0) {
        std::fill(pos_zone_stamp_.begin(), pos_zone_stamp_.end(), 0u);
        pos_zone_epoch_ = 1;
    }
    arc_lap_len = queue_len;
    arc_scan_pos = 0;
    arc_probe_used = 0;
    arc_pointer = in_cache.q.head;
    zone_samples.fill(0);
    zone_evictions.fill(0);
}

void TLCacheMABCache::arm_zone_remember(uint32_t pos, int zone) {
    if (zone < 0 || zone >= N_ZONES) return;
    if (pos >= pos_zone_.size()) return;
    pos_zone_[pos] = static_cast<uint8_t>(zone);
    pos_zone_stamp_[pos] = pos_zone_epoch_;
}

int TLCacheMABCache::arm_zone_of_pos(uint32_t pos) const {
    if (pos >= pos_zone_.size()) return -1;
    if (pos_zone_stamp_[pos] != pos_zone_epoch_) return -1;
    return static_cast<int>(pos_zone_[pos]);
}

void TLCacheMABCache::arm_sample_arc(std::vector<uint32_t>& sampled_objects,
                                     uint32_t& steps_out) {
    const uint32_t L = static_cast<uint32_t>(in_cache.metas.size());
    steps_out = 0;
    sampled_objects.clear();
    if (L == 0 || in_cache.q.head >= L) return;

    // Restart the lap if there is none, if the cursor went stale because its
    // slot was evicted and reused, or if the queue changed size under us.
    if (arc_lap_len == 0 || arc_lap_len > L || arc_pointer >= L ||
        L > arc_lap_len * 2) {
        arm_zone_begin_lap(L);
    }

    uint32_t lo_i = 0, w_i = 1, half_i = 0;
    auto load_arc_bounds = [&]() {
        lo_i = static_cast<uint32_t>(arc_lo * arc_lap_len) % arc_lap_len;
        w_i = static_cast<uint32_t>(arc_width * arc_lap_len);
        if (w_i == 0) w_i = 1;
        if (w_i > arc_lap_len) w_i = arc_lap_len;
        half_i = w_i / 2;
    };
    load_arc_bounds();

    // Outside the arc only the frequency probe fires, copied from
    // TLCacheCache::rank(), so the outside stays measured for the lap rule
    // without becoming a second sampling region.
    const uint32_t probe_quota =
        std::max<uint32_t>(1u, static_cast<uint32_t>(sample_rate) / 8u);
    // One call must never be able to grind through a whole lap.
    const uint32_t step_cap =
        std::max<uint32_t>(1024u, static_cast<uint32_t>(sample_rate) * 4u);

    sampled_objects.reserve(sample_rate);
    uint32_t steps = 0;
    while (steps < step_cap &&
           static_cast<uint32_t>(sampled_objects.size()) < sample_rate) {
        if (arc_scan_pos >= arc_lap_len) {
            // The original closes the lap and carries on inside the same
            // call, which is what keeps it from ever returning nothing.
            arm_arc_lap_end();
            load_arc_bounds();
            continue;
        }
        const uint32_t pos = arc_pointer;
        if (pos >= L) break;
        const uint32_t t = (arc_scan_pos >= lo_i)
                               ? (arc_scan_pos - lo_i)
                               : (arc_scan_pos + arc_lap_len - lo_i);
        int zone = ZONE_OUT;
        bool take = false;
        if (t < w_i) {
            // Dense inside the arc, the way the original samples every object
            // inside its window. This is what keeps the cost per sample flat.
            zone = (t < half_i) ? ZONE_LEAD : ZONE_TRAIL;
            take = true;
        } else if (arc_probe_used < probe_quota) {
            const uint16_t freq =
                static_cast<uint16_t>(in_cache.metas[pos]._freq - 1);
            if (freq < sample_boundary) {
                take = true;
                arc_probe_used++;
            }
        }
        if (take) {
            sampled_objects.emplace_back(pos);
            zone_samples[static_cast<size_t>(zone)]++;
            arm_zone_remember(pos, zone);
        }
        arc_pointer = in_cache.dq[pos].next;
        arc_scan_pos++;
        steps++;
    }

    if (sampled_objects.empty()) {
        // The original never hands back an empty pool. If the cursor is deep
        // outside the arc with nothing worth probing, take the slots it is
        // standing on so eviction can still make progress.
        const uint32_t want = std::max<uint32_t>(1u, sample_rate / 4u);
        while (static_cast<uint32_t>(sampled_objects.size()) < want &&
               arc_scan_pos < arc_lap_len && arc_pointer < L) {
            const uint32_t pos = arc_pointer;
            sampled_objects.emplace_back(pos);
            zone_samples[ZONE_OUT]++;
            arm_zone_remember(pos, ZONE_OUT);
            arc_pointer = in_cache.dq[pos].next;
            arc_scan_pos++;
            steps++;
        }
    }
    steps_out = steps;
}

void TLCacheMABCache::arm_arc_lap_end() {
    // Both nudges are the sampling_lru test from TLCacheCache::rank():
    //   evcition_distribution[2] * [1] > [0] * [3]  ->  sampling_lru++
    //   else if (sampling_lru > 1)                  ->  sampling_lru--
    // which is a cross multiplication of two evictions-per-sample rates
    // followed by a one unit move of the boundary. Here it runs twice: once
    // on inside versus outside the arc to size it, once on the arc's own two
    // halves to slide it. Nudging both edges leaves a middle arc in the
    // middle, which a single global boundary cannot express.

    // Explore is the control arm: it covers the whole ring on purpose and is
    // left alone, so there is always one arm that did not adapt.
    if (start_arc_width >= 1.0) {
        arm_zone_begin_lap(static_cast<uint32_t>(in_cache.metas.size()));
        return;
    }

    const uint64_t lead_e = zone_evictions[ZONE_LEAD];
    const uint64_t lead_s = zone_samples[ZONE_LEAD];
    const uint64_t trail_e = zone_evictions[ZONE_TRAIL];
    const uint64_t trail_s = zone_samples[ZONE_TRAIL];
    const uint64_t in_e = lead_e + trail_e;
    const uint64_t in_s = lead_s + trail_s;
    const uint64_t out_e = zone_evictions[ZONE_OUT];
    const uint64_t out_s = zone_samples[ZONE_OUT];

    bool moved = false;
    if (in_s > 0) {
        // Note the test runs even when the outside produced no samples. That
        // is deliberate and is what the original does: with no snapshot,
        // evcition_distribution[2] and [3] stay zero, the cross product is
        // false and sampling_lru decrements. Without it an arc that swallows
        // the lap has no outside left to compare against and can never
        // shrink again.
        if (in_e * out_s > out_e * in_s) {
            arc_width += ARC_UNIT;
            moved = true;
        } else {
            arc_width -= ARC_UNIT;
            moved = true;
        }
    }
    if (lead_s > 0 && trail_s > 0) {
        // Lead is the older side of the arc, so favouring it walks the arc
        // toward q.head.
        arc_lo += (lead_e * trail_s > trail_e * lead_s) ? -ARC_UNIT : ARC_UNIT;
        moved = true;
    }

    // Leash. One unit per lap is a random walk when the two rates are close,
    // and over a few hundred laps that is enough to turn any arm into any
    // other. The arm keeps its identity; the nudge only tunes it locally.
    const double w_lo =
        std::max(ARC_MIN_WIDTH, start_arc_width * (1.0 - ARC_WIDTH_SPAN));
    const double w_hi =
        std::min(ARC_MAX_WIDTH, start_arc_width * (1.0 + ARC_WIDTH_SPAN));
    if (arc_width < w_lo) arc_width = w_lo;
    if (arc_width > w_hi) arc_width = w_hi;
    double drift = arc_lo - start_arc_lo;
    if (drift > 0.5) drift -= 1.0;
    if (drift < -0.5) drift += 1.0;
    if (drift > ARC_MAX_DRIFT) drift = ARC_MAX_DRIFT;
    if (drift < -ARC_MAX_DRIFT) drift = -ARC_MAX_DRIFT;
    arc_lo = start_arc_lo + drift;
    if (arc_lo < 0.0) arc_lo += 1.0;
    if (arc_lo >= 1.0) arc_lo -= 1.0;
    if (moved) arm_slider_steps++;

    arm_zone_begin_lap(static_cast<uint32_t>(in_cache.metas.size()));
}

void TLCacheMABCache::arm_begin_decision_segment() {
    arm_selected_requests = 0;
    arm_selected_bytes = 0;
    arm_selected_miss_bytes = 0;
    arm_selected_shadow_miss_bytes = 0;
    arm_recheck_requests = 0;
    arm_recheck_bytes = 0;
    arm_recheck_miss_bytes = 0;
    arm_recheck_shadow_miss_bytes = 0;
    arm_recheck_strikes = 0;
}

void TLCacheMABCache::arm_selector_begin_regime() {
    zone_samples.fill(0);
    zone_evictions.fill(0);
    arm_context_ready = true;
    bool matched = false;
    const uint8_t arm = profiler.select_regime_arm(&matched);
    arm_set_arc_from_arm(arm);
    arm_phase = ArmPhase::Committed;
    policy_frozen = true;
    if (matched) inject_count++;
}

#if 0
void TLCacheMABCache::arm_maybe_redecide() {
    if (arm_recheck_exhausted) return;
    if (arm_recheck_requests < ARM_RECHECK_REQUESTS) return;

    const double chunk_delta = (arm_recheck_bytes > 0)
        ? (static_cast<double>(arm_recheck_shadow_miss_bytes) -
           static_cast<double>(arm_recheck_miss_bytes)) /
          static_cast<double>(arm_recheck_bytes)
        : 0.0;
    arm_recheck_requests = 0;
    arm_recheck_bytes = 0;
    arm_recheck_miss_bytes = 0;
    arm_recheck_shadow_miss_bytes = 0;

    if (chunk_delta > -ARM_RECHECK_MIN_LOSS) {
        arm_recheck_strikes = 0;
        return;
    }
    if (++arm_recheck_strikes < ARM_RECHECK_STRIKES) return;

    // Close this arm's segment so its realized delta is attributed to the arm
    // that produced it, then pick again from refreshed cache state.
    arm_write_model_event("arm_recheck");
    arm_failed_mask |= static_cast<uint8_t>(1u << committed_arm);
    arm_reselect_count++;

    arm_build_context();
    uint8_t excluded = arm_failed_mask;
    if (arm_failed_mask == 0x0F) {
        // Nothing better is on offer: take the least-bad arm and stop
        // second-guessing it for the rest of this regime.
        excluded = 0;
        arm_recheck_exhausted = true;
    }
    committed_arm = arm_model_choose(excluded);
    arm_phase = ArmPhase::Committed;
    policy_frozen = true;
    arm_model_prediction_ready = true;
    arm_begin_decision_segment();

    if (diag.enabled()) {
        const double cum_miss = (total_req_count > 0)
            ? (double)total_miss_count / (double)total_req_count : 0.0;
        std::ostringstream detail;
        detail << "arm=" << (int)committed_arm
               << ";chunk_delta=" << chunk_delta
               << ";failed_mask=" << (int)arm_failed_mask;
        diag.log_event(global_seq, "arm_reselect", detail.str().c_str(),
                       ema_miss_rate, cum_miss, mab_weights);
    }
}

void TLCacheMABCache::arm_selector_score_request(
    uint32_t size, bool is_miss, bool shadow_is_miss) {
    if (!arm_context_ready) return;
    if (arm_selector_mode == ArmSelectorMode::LEGACY) return;
    // Whole window vs shadow. The slider may move shares; the label is
    // still this start + this window, not each ±1 step.
    arm_selected_requests++;
    arm_selected_bytes += size;
    if (is_miss) arm_selected_miss_bytes += size;
    if (shadow_is_miss) arm_selected_shadow_miss_bytes += size;
}

bool TLCacheMABCache::arm_probe_complete() const {
    if (arm_probe_block < ARM_PROBE_BLOCKS) return false;
    for (const auto& outcome : arm_probe_outcomes) {
        if (outcome.requests < ARM_PROBE_MIN_REQUESTS ||
            outcome.bytes == 0) return false;
    }
    return true;
}

void TLCacheMABCache::arm_write_training_rows() {
    if (!arm_model_prediction_ready || arm_selected_bytes == 0) return;
    ensure_parent_directory(arm_training_file);
    const bool write_header =
        !std::filesystem::exists(arm_training_file) ||
        std::filesystem::file_size(arm_training_file) == 0;
    std::ofstream file(arm_training_file, std::ios::app);
    if (!file) throw std::runtime_error("cannot append " + arm_training_file);
    if (write_header) {
        file << "trace,cache_size,regime_id,arm,probe_order,requests,bytes,"
                "miss_bytes,shadow_miss_bytes,delta_bmr,confidence";
        for (int i = 0; i < ARM_MODEL_FEATURES; ++i) file << ",f_" << i;
        file << ",cell_id,start_arc_lo,start_arc_width,"
                "end_arc_lo,end_arc_width\n";
    }
    const double delta =
        (static_cast<double>(arm_selected_shadow_miss_bytes) -
         static_cast<double>(arm_selected_miss_bytes)) /
        static_cast<double>(arm_selected_bytes);
    const double confidence =
        std::min(1.0, static_cast<double>(arm_selected_requests) / 1024.0);
    file << std::setprecision(17)
         << csv_quote(arm_trace_id) << ","
         << csv_quote(arm_cache_size_label) << ","
         << regime_id << "," << static_cast<int>(start_arm) << ","
         << arm_cell_id << ","
         << arm_selected_requests << "," << arm_selected_bytes << ","
         << arm_selected_miss_bytes << "," << arm_selected_shadow_miss_bytes
         << "," << delta << "," << confidence;
    for (double value : arm_model_row(start_arm)) file << "," << value;
    file << "," << arm_cell_id;
    file << "," << start_arc_lo << "," << start_arc_width
         << "," << arc_lo << "," << arc_width << "\n";
    previous_selector_arm = start_arm;
    previous_selector_delta = delta;
}

void TLCacheMABCache::arm_probe_commit() {
    if (!arm_probe_complete()) return;
    int best = 0;
    double best_delta = -1e300;
    for (int arm = 0; arm < 4; ++arm) {
        const auto& outcome = arm_probe_outcomes[static_cast<size_t>(arm)];
        const double delta =
            (static_cast<double>(outcome.shadow_miss_bytes) -
             static_cast<double>(outcome.miss_bytes)) /
            static_cast<double>(outcome.bytes);
        if (delta > best_delta) {
            best_delta = delta;
            best = arm;
        }
    }
    arm_write_training_rows();
    committed_arm = static_cast<uint8_t>(best);
    previous_selector_arm = best;
    previous_selector_delta = best_delta;
    arm_phase = ArmPhase::Committed;
    for (int arm = 0; arm < mab_k; ++arm) {
        mab_weights[arm] = (arm == best) ? 1.0 : 1e-3;
    }
    mab_normalize_weights();
    mab_compute_probs();
}

uint8_t TLCacheMABCache::arm_model_choose(uint8_t excluded_mask) {
    int best = -1;
    double best_score = -1e300;
    for (int arm = 0; arm < 4; ++arm) {
        const std::vector<double> row =
            arm_model_row(static_cast<uint8_t>(arm));
        arm_pred_mean[static_cast<size_t>(arm)] =
            arm_model_predict(arm_mean_booster, row);
        arm_pred_downside[static_cast<size_t>(arm)] =
            arm_model_predict(arm_downside_booster, row);
        if (excluded_mask & static_cast<uint8_t>(1u << arm)) continue;
        const double score =
            (1.0 - arm_risk_lambda) *
                arm_pred_mean[static_cast<size_t>(arm)] +
            arm_risk_lambda *
                arm_pred_downside[static_cast<size_t>(arm)];
        if (best < 0 || score > best_score) {
            best_score = score;
            best = arm;
        }
    }
    if (best < 0) best = 0; // every arm excluded: keep a valid selection
    for (int arm = 0; arm < mab_k; ++arm) {
        mab_weights[arm] = (arm == best) ? 1.0 : 1e-3;
    }
    mab_normalize_weights();
    mab_compute_probs();
    previous_selector_arm = best;
    return static_cast<uint8_t>(best);
}

void TLCacheMABCache::arm_write_model_event(const char* reason) {
    if (!arm_model_prediction_ready || arm_selected_bytes == 0) return;
    ensure_parent_directory(arm_model_events_file);
    const bool write_header =
        !std::filesystem::exists(arm_model_events_file) ||
        std::filesystem::file_size(arm_model_events_file) == 0;
    std::ofstream file(arm_model_events_file, std::ios::app);
    if (!file) throw std::runtime_error("cannot append " + arm_model_events_file);
    if (write_header) {
        file << "trace,cache_size,regime_id,selected_arm,requests,bytes,"
                "miss_bytes,shadow_miss_bytes,delta_bmr,close_reason";
        for (int arm = 0; arm < 4; ++arm) file << ",mean_" << arm;
        for (int arm = 0; arm < 4; ++arm) file << ",downside_" << arm;
        for (int i = 0; i < ARM_CONTEXT_FEATURES; ++i) file << ",f_" << i;
        file << "\n";
    }
    const double delta =
        (static_cast<double>(arm_selected_shadow_miss_bytes) -
         static_cast<double>(arm_selected_miss_bytes)) /
        static_cast<double>(arm_selected_bytes);
    file << std::setprecision(17)
         << csv_quote(arm_trace_id) << ","
         << csv_quote(arm_cache_size_label) << "," << regime_id
         << "," << static_cast<int>(committed_arm) << ","
         << arm_selected_requests << "," << arm_selected_bytes << ","
         << arm_selected_miss_bytes << "," << arm_selected_shadow_miss_bytes
         << "," << delta << "," << csv_quote(reason ? reason : "unknown");
    for (double value : arm_pred_mean) file << "," << value;
    for (double value : arm_pred_downside) file << "," << value;
    for (double value : arm_regime_context) file << "," << value;
    file << "\n";
    previous_selector_delta = delta;
}
#endif

void TLCacheMABCache::arm_selector_finish_regime(const char* reason) {
    (void)reason;
    arm_context_ready = false;
}

void TLCacheMABCache::diag_maybe_snapshot(const char* reason_event) {
    if (!diag.enabled()) return;

    const double cum_miss = (total_req_count > 0)
        ? (double)total_miss_count / (double)total_req_count
        : 0.0;

    diag.log_timeseries(global_seq,
                        profiler_mode_cstr(),
                        last_mode_arm,
                        ema_miss_rate,
                        last_window_miss,
                        cum_miss,
                        miss_rate_slope,
                        mab_weights,
                        mab_probs,
                        mab_arm_eviction_count,
                        mab_arm_select_count);

    if (reason_event) {
        diag.log_event(global_seq, reason_event, profiler_mode_cstr(),
                       ema_miss_rate, cum_miss, mab_weights);
    }
}

void TLCacheMABCache::init_with_params(const map<string, string> &params) {
    TLCacheCache::init_with_params(params);

    global_seq        = 0;
    ema_miss_rate     = 0.0;
    miss_rate_slope   = 0.0;
    last_window_miss  = 0.0;
    total_req_count   = 0;
    total_miss_count  = 0;
    window_req_count  = 0;
    window_miss_count = 0;
    last_mode_arm     = 0;
    mab_rng_state     = 0xC0FFEEULL ^ (uint64_t)this->getSize();
    diag_enable       = false;
    diag_interval     = 10000;
    diag_prefix       = "mab_diag";
    train_log_enable  = true;
    train_log_interval = 10000;
    train_log_prefix  = "mab_train";
    mab_off           = false;
    total_bytes_req   = 0;
    total_bytes_miss  = 0;
    total_shadow_misses = 0;
    total_shadow_bytes_miss = 0;
    regime_id         = 0;
    inject_count      = 0;
    arm_context_ready = false;

    auto get_param = [&](const string &a, const string &b) -> map<string,string>::const_iterator {
        auto it = params.find(a);
        if (it == params.end()) it = params.find(b);
        return it;
    };

    auto p_it = get_param("policy_file", "policy-file");
    if (p_it != params.end()) {
        policy_file_path = p_it->second;
    }

    auto c_it = get_param("config_file", "config-file");
    if (c_it != params.end()) {
        config_file_path = c_it->second;
    }

    arm_strategy = ArmStrategy::MODE;
    mab_gamma = 0.05;

    for (auto &it : params) {
        if (it.first == "arm_count" || it.first == "arm-count") {
            int k = stoi(it.second);
            if (k < 1) k = 1;
            if (k > (int)MAX_ARMS) k = (int)MAX_ARMS;
            mab_k = (uint8_t)k;
        } else if (it.first == "arm_strategy" || it.first == "arm-strategy") {
            if (it.second == "mode")            arm_strategy = ArmStrategy::MODE;
            else if (it.second == "position")   arm_strategy = ArmStrategy::POSITION;
            else if (it.second == "frequency")  arm_strategy = ArmStrategy::FREQUENCY;
            else if (it.second == "age")        arm_strategy = ArmStrategy::AGE;
        } else if (it.first == "mab_gamma" || it.first == "mab-gamma") {
            mab_gamma = stod(it.second);
            if (mab_gamma <= 0.0) mab_gamma = 0.01;
            if (mab_gamma >= 1.0) mab_gamma = 0.99;
        } else if (it.first == "diag_enable" || it.first == "diag-enable") {
            diag_enable = (it.second == "1" || it.second == "true" || it.second == "yes");
        } else if (it.first == "diag_interval" || it.first == "diag-interval") {
            diag_interval = (uint64_t)stoull(it.second);
            if (diag_interval == 0) diag_interval = 10000;
        } else if (it.first == "diag_prefix" || it.first == "diag-prefix") {
            diag_prefix = it.second;
        } else if (it.first == "train_log_enable" || it.first == "train-log-enable") {
            train_log_enable = (it.second == "1" || it.second == "true" || it.second == "yes");
        } else if (it.first == "train_log_interval" || it.first == "train-log-interval") {
            train_log_interval = (uint64_t)stoull(it.second);
            if (train_log_interval == 0) train_log_interval = 10000;
        } else if (it.first == "train_log_prefix" || it.first == "train-log-prefix") {
            train_log_prefix = it.second;
        } else if (it.first == "mab_off" || it.first == "mab-off") {
            mab_off = (it.second == "1" || it.second == "true" || it.second == "yes");
        } else if (it.first == "arm_selector" || it.first == "arm-selector") {
            if (it.second != "knn" && it.second != "legacy") {
                throw std::runtime_error(
                    "first-stage LightGBM removed; use arm-selector=knn");
            }
        }
    }

    // Mode strategy uses six start mixes (four original + lean middle/ends).
    if (arm_strategy == ArmStrategy::MODE && mab_k != MODE_ARM_COUNT) {
        mab_k = MODE_ARM_COUNT;
    }

    for (int i = 0; i < MAX_ARMS; i++) {
        mab_weights[i]            = 1.0;
        mab_probs[i]              = 1.0 / mab_k;
        mab_arm_samples[i]        = 0;
        mab_arm_eviction_count[i] = 0;
        mab_arm_select_count[i]   = 0;
        arm_regime_selects[i]     = 0;
    }
    mab_reward_max      = 1.0;
    mab_total_evictions = 0;
    arm_phase           = ArmPhase::Investigate;
    arm_gap_hold        = 0;
    committed_arm       = 0;
    rr_probe_idx        = 0;
    policy_frozen       = false;
    arm_probe_block     = 0;
    arm_probe_selects_in_block = 0;
    arm_probe_active    = false;
    arm_selected_requests = 0;
    arm_selected_bytes = 0;
    arm_selected_miss_bytes = 0;
    arm_selected_shadow_miss_bytes = 0;
    for (auto& outcome : arm_probe_outcomes) outcome = ArmProbeOutcome{};
    mab_compute_probs();

    string mode_str = "train";
    auto mode_it = get_param("profiler_mode", "profiler-mode");
    if (mode_it != params.end()) {
        mode_str = mode_it->second;
    }
    string feature_mask = "111111111111111";
    auto mask_it = get_param("feature_mask", "feature-mask");
    if (mask_it != params.end()) {
        feature_mask = mask_it->second;
    }
    bool persist_event_logs = true;
    auto event_it = get_param("event_logs", "event-logs");
    if (event_it != params.end()) {
        persist_event_logs = (event_it->second == "1" ||
                              event_it->second == "true" ||
                              event_it->second == "yes");
    }

    // Auto-enable diagnostics for TEST unless explicitly disabled.
    if (!params.count("diag_enable") && !params.count("diag-enable")) {
        if (mode_str == "test") diag_enable = true;
    }
    if (diag_prefix == "mab_diag") {
        diag_prefix = (mode_str == "test") ? "mab_diag_test" : "mab_diag_train";
    }

    // Training logger CSV (progress/learning) is retired. Regime/feature stats
    // are written separately under cdt_logs/train/.
    if (!params.count("train_log_enable") && !params.count("train-log-enable")) {
        train_log_enable = false;
    }

    // A frozen run is a measurement reference only: it must never learn,
    // inject, or write back to the policy matrix.
    if (mab_off) {
        mode_str = "test";
        train_log_enable = false;
    }

    profiler.init(mode_str, policy_file_path, config_file_path, this->getSize(), mab_k,
                  train_log_enable, train_log_prefix, train_log_interval,
                  feature_mask, persist_event_logs);
    diag.init(diag_enable, diag_prefix, diag_interval, mab_k);
    if (diag.enabled()) {
        diag.log_event(0, "start", mode_str.c_str(), 0.0, 0.0, mab_weights);
    }
}

bool TLCacheMABCache::lookup(const SimpleRequest &req) {
    bool out_cache_hit = false;
    auto it = key_map.find(req.id);
    if (it != key_map.end() && it->second.list_idx == 1) {
        out_cache_hit = true;
    }

    bool main_hit = TLCacheCache::lookup(req);
    bool is_miss = !main_hit;

    total_req_count++;
    if (is_miss) total_miss_count++;

    total_bytes_req += (uint64_t)req.size;
    if (is_miss) total_bytes_miss += (uint64_t)req.size;

    if (!current_shadow_hit) {
        total_shadow_misses++;
        total_shadow_bytes_miss += (uint64_t)req.size;
    }

    window_req_count++;
    if (is_miss) {
        window_miss_count++;
    }

    if (window_req_count >= 1000) {
        double current_window_miss_rate = (double)window_miss_count / window_req_count;
        last_window_miss = current_window_miss_rate;
        if (global_seq > 1000) {
            miss_rate_slope = current_window_miss_rate - ema_miss_rate;
            ema_miss_rate = 0.8 * ema_miss_rate + 0.2 * current_window_miss_rate;
        } else {
            ema_miss_rate = current_window_miss_rate;
            miss_rate_slope = 0.0;
        }
        window_req_count = 0;
        window_miss_count = 0;
    }

    if (mab_off) {
        global_seq++;
        return main_hit;
    }

    std::vector<double> current_weights(mab_weights, mab_weights + mab_k);
    std::vector<double> new_weights;
    bool regime_reset = false;

    double req_timestamp = (current_req_clock_time > 0.0) ? current_req_clock_time : (double)global_seq;
    bool req_is_write = current_req_is_write;

    bool should_update_weights = profiler.add_request(
        req.id,
        req.size,
        req_is_write,
        req_timestamp,
        out_cache_hit,
        is_miss,
        !current_shadow_hit,
        current_weights,
        (arm_phase == ArmPhase::Committed),
        new_weights,
        &regime_reset
    );

    if (regime_reset) {
        const std::string& close_reason = profiler.get_last_regime_close_reason();
        arm_selector_finish_regime(
            close_reason.empty() ? "regime_close" : close_reason.c_str());
        regime_id++;
        // Feature ε closed/opened a regime. Restart arm investigation unless
        // TEST immediately injects the closest stored policy below.
        if (!should_update_weights) {
            mab_on_regime_reset();
            policy_frozen = false;
            if (diag.enabled()) {
                const double cum_miss = (total_req_count > 0)
                    ? (double)total_miss_count / (double)total_req_count
                    : 0.0;
                diag.log_event(global_seq, "regime_reset", "investigate",
                               ema_miss_rate, cum_miss, mab_weights);
            }
        }
        arm_selector_begin_regime();
    }

    if (should_update_weights && !new_weights.empty()) {
        std::ostringstream detail;
        detail << "inject";
        for (int i = 0; i < mab_k && i < (int)new_weights.size(); i++) {
            detail << ";w" << i << "=" << new_weights[i];
            mab_weights[i] = new_weights[i];
        }
        mab_normalize_weights();
        mab_compute_probs();
        inject_count++;
        // Held-out replay: freeze Exp3 on the injected mix.
        policy_frozen = true;
        arm_phase = ArmPhase::Committed;
        committed_arm = 0;
        for (int i = 1; i < mab_k; i++) {
            if (mab_weights[i] > mab_weights[committed_arm]) committed_arm = (uint8_t)i;
        }

        if (diag.enabled()) {
            const double cum_miss = (total_req_count > 0)
                ? (double)total_miss_count / (double)total_req_count
                : 0.0;
            diag.log_event(global_seq, "policy_inject", detail.str().c_str(),
                           ema_miss_rate, cum_miss, mab_weights);
        }
    }

    if (!arm_context_ready) {
        arm_selector_begin_regime();
    }

    global_seq++;
    if (diag.enabled() && diag.interval() > 0 &&
        (global_seq % diag.interval()) == 0) {
        diag_maybe_snapshot(nullptr);
    }
    return main_hit;
}

double TLCacheMABCache::mab_next_rand() {
    // xorshift64*
    mab_rng_state ^= mab_rng_state >> 12;
    mab_rng_state ^= mab_rng_state << 25;
    mab_rng_state ^= mab_rng_state >> 27;
    uint64_t r = mab_rng_state * 2685821657736338717ULL;
    return (r >> 11) * (1.0 / 9007199254740992.0); // [0,1)
}

uint8_t TLCacheMABCache::mab_select_arm() {
    if (arm_phase == ArmPhase::Committed) {
        // TEST replays the injected row's own arm. Sampling the mix instead
        // made every far match a lottery, which is what the table already
        // encodes as a near one-hot choice.
        return committed_arm < mab_k ? committed_arm : 0;
    }

    // Investigate: round-robin so each mode gets fair credit before commit.
    uint8_t arm = (uint8_t)(rr_probe_idx % mab_k);
    rr_probe_idx++;
    return arm;
}

double TLCacheMABCache::mab_effective_gamma() const {
    if (arm_phase == ArmPhase::Committed) return COMMIT_GAMMA;
    return INVESTIGATE_GAMMA;
}

void TLCacheMABCache::mab_on_regime_reset() {
    for (int i = 0; i < mab_k; i++) {
        mab_weights[i] = 1.0;
        arm_regime_selects[i] = 0;
    }
    arm_phase = ArmPhase::Investigate;
    arm_gap_hold = 0;
    committed_arm = 0;
    rr_probe_idx = 0;
    arm_probe_block = 0;
    arm_probe_selects_in_block = 0;
    arm_probe_active = false;
    mab_normalize_weights();
    mab_compute_probs();
}

void TLCacheMABCache::mab_maybe_commit() {
    if (arm_phase != ArmPhase::Investigate) return;
    if (arm_strategy != ArmStrategy::MODE) return;

    for (int i = 0; i < mab_k; i++) {
        if (arm_regime_selects[i] < ARM_N_MIN) return;
    }

    int best = 0, second = -1;
    for (int i = 1; i < mab_k; i++) {
        if (mab_weights[i] > mab_weights[best]) {
            second = best;
            best = i;
        } else if (second < 0 || mab_weights[i] > mab_weights[second]) {
            second = i;
        }
    }
    if (second < 0) second = best;

    double sum_w = 0.0;
    for (int i = 0; i < mab_k; i++) sum_w += mab_weights[i];
    if (sum_w <= 0.0) return;

    double gap = (mab_weights[best] - mab_weights[second]) / sum_w;
    if (gap >= ARM_GAP) {
        arm_gap_hold++;
    } else {
        arm_gap_hold = 0;
    }

    if (arm_gap_hold < ARM_GAP_HOLD) return;

    arm_phase = ArmPhase::Committed;
    committed_arm = (uint8_t)best;
    // Lock almost all mass on the winner; keep tiny floor for numerical safety.
    for (int i = 0; i < mab_k; i++) {
        mab_weights[i] = (i == best) ? 1.0 : 1e-3;
    }
    mab_normalize_weights();
    mab_compute_probs();

    if (diag.enabled()) {
        const double cum_miss = (total_req_count > 0)
            ? (double)total_miss_count / (double)total_req_count
            : 0.0;
        std::ostringstream detail;
        detail << "arm=" << (int)committed_arm << ";gap=" << gap;
        diag.log_event(global_seq, "arm_commit", detail.str().c_str(),
                       ema_miss_rate, cum_miss, mab_weights);
    }
}

void TLCacheMABCache::mab_sample_mode(uint8_t mode, vector<uint32_t> &sampled_objects,
                                      uint32_t &steps_out) {
    const uint32_t queue_len = (uint32_t)in_cache.metas.size();
    uint32_t pos = in_cache.q.head;
    uint32_t steps = 0;

    if (queue_len == 0 || pos >= queue_len) {
        steps_out = 0;
        return;
    }

    const uint8_t mode_id = (uint8_t)(mode % 4);

    // Each mode owns a genuinely different slice of the queue, and we
    // STRIDE across that slice instead of taking a contiguous prefix. With
    // sample_rate ~= 1% of the queue, a contiguous scan never left the head,
    // so the arms were near-identical. Strided regions fix that.
    //
    // Index 0 = head (newest), high index = tail (aged).
    uint32_t lo1 = 0, hi1 = queue_len;   // primary region
    uint32_t lo2 = 0, hi2 = 0;           // optional second region (Balanced)
    switch (mode_id) {
    case 0: // Balanced: split budget between head quarter and tail quarter.
        lo1 = 0;                       hi1 = std::max(1u, queue_len / 4);
        lo2 = queue_len - std::max(1u, queue_len / 4); hi2 = queue_len;
        break;
    case 1: // Tail+: aged / unpopular half.
        lo1 = queue_len / 2;           hi1 = queue_len;
        break;
    case 2: // Head+: newly admitted half.
        lo1 = 0;                       hi1 = std::max(1u, queue_len / 2);
        break;
    case 3: // Explore: whole queue, uniform stride.
    default:
        lo1 = 0;                       hi1 = queue_len;
        break;
    }

    sampled_objects.clear();
    sampled_objects.reserve(sample_rate);

    const uint32_t budget1 = (hi2 > lo2) ? (sample_rate / 2 + (sample_rate & 1u)) : sample_rate;
    const uint32_t budget2 = (hi2 > lo2) ? (sample_rate / 2) : 0;

    auto region_len = [](uint32_t lo, uint32_t hi) -> uint32_t {
        return (hi > lo) ? (hi - lo) : 0u;
    };
    const uint32_t len1 = region_len(lo1, hi1);
    const uint32_t len2 = region_len(lo2, hi2);
    const uint32_t stride1 = (budget1 > 0 && len1 > budget1) ? (len1 / budget1) : 1u;
    const uint32_t stride2 = (budget2 > 0 && len2 > budget2) ? (len2 / budget2) : 1u;

    // Rotate the stride phase each rank() so Explore/others cover fresh objects.
    const uint32_t phase = (uint32_t)(global_seq % (stride1 > 0 ? stride1 : 1u));

    uint32_t got1 = 0, got2 = 0;
    while (steps < queue_len && pos < queue_len &&
           (uint32_t)sampled_objects.size() < sample_rate) {
        bool take = false;
        if (steps >= lo1 && steps < hi1 && got1 < budget1) {
            if (((steps - lo1 + phase) % stride1) == 0u) { take = true; got1++; }
        }
        if (!take && len2 > 0 && steps >= lo2 && steps < hi2 && got2 < budget2) {
            if (((steps - lo2) % stride2) == 0u) { take = true; got2++; }
        }
        if (take) sampled_objects.emplace_back(pos);
        pos = in_cache.dq[pos].next;
        steps++;
    }

    // Fill remainder so prediction still has enough candidates.
    if ((uint32_t)sampled_objects.size() < sample_rate) {
        vector<bool> seen(queue_len, false);
        for (uint32_t p : sampled_objects) {
            if (p < queue_len) seen[p] = true;
        }
        pos = in_cache.q.head;
        for (uint32_t s = 0; s < queue_len && (uint32_t)sampled_objects.size() < sample_rate; s++) {
            if (pos < queue_len && !seen[pos]) {
                sampled_objects.emplace_back(pos);
                seen[pos] = true;
            }
            pos = in_cache.dq[pos].next;
        }
        steps = queue_len;
    }

    steps_out = steps;
}

uint32_t TLCacheMABCache::rank() {
    if (mab_off || !booster || in_cache.metas.empty()) {
        return TLCacheCache::rank();
    }
    // Held-out TEST replays the closest training policy and never explores or
    // updates Exp3. An empty/unusable table remains on the baseline path.
    if (profiler.get_mode() == ProfilerMode::TEST && !policy_frozen &&
        !arm_slider_active()) {
        return TLCacheCache::rank();
    }

    vector<uint32_t> sampled_objects;

    if (initial_queue_length == 0)
        initial_queue_length = in_cache.metas.size();

    sample_rate = 1024;
    if (sample_rate >= initial_queue_length * 0.01 + eviction_rate)
        sample_rate = initial_queue_length > 2
                    ? (uint16_t)(initial_queue_length * 0.01 + eviction_rate)
                    : 2;

    mab_compute_probs();

    uint32_t steps = 0;
    const uint32_t queue_len = (uint32_t)in_cache.metas.size();
    uint32_t pos = in_cache.q.head;
    if (pos >= queue_len) return TLCacheCache::rank();

    if (arm_slider_active()) {
        if (!arm_context_ready) arm_selector_begin_regime();
        last_mode_arm = start_arm;
        if (last_mode_arm < MAX_ARMS) {
            mab_arm_select_count[last_mode_arm]++;
        }
        // Lap bookkeeping and the edge nudge both live inside the sampler,
        // because the persistent cursor is what defines a lap now.
        arm_sample_arc(sampled_objects, steps);
    } else if (arm_strategy == ArmStrategy::MODE) {
        last_mode_arm = mab_select_arm();
        if (last_mode_arm < MAX_ARMS) {
            mab_arm_select_count[last_mode_arm]++;
            if (arm_phase == ArmPhase::Investigate) {
                arm_regime_selects[last_mode_arm]++;
            }
        }
        mab_sample_mode(last_mode_arm, sampled_objects, steps);
    } else {
        // Legacy ablation path: soft mixture over position/age/freq buckets.
        mab_allocate_samples();
        uint32_t arm_collected[MAX_ARMS] = {0};
        sampled_objects.reserve(sample_rate);

        int arms_remaining = 0;
        for (int i = 0; i < mab_k; i++)
            if (mab_arm_samples[i] > 0) arms_remaining++;

        while (steps < queue_len && arms_remaining > 0
               && (uint32_t)sampled_objects.size() < sample_rate && pos < queue_len) {
            uint8_t arm = mab_get_arm(pos);
            if (arm_collected[arm] < mab_arm_samples[arm]) {
                sampled_objects.emplace_back(pos);
                arm_collected[arm]++;
                if (arm_collected[arm] == mab_arm_samples[arm])
                    arms_remaining--;
            }
            pos = in_cache.dq[pos].next;
            steps++;
        }

        uint32_t remaining = sample_rate - (uint32_t)sampled_objects.size();
        if (remaining > 0 && queue_len > (uint32_t)sampled_objects.size()) {
            vector<bool> sampled_flag(queue_len, false);
            for (uint32_t p : sampled_objects) {
                if (p < queue_len) sampled_flag[p] = true;
            }
            pos = in_cache.q.head;
            for (uint32_t s = 0; s < queue_len && remaining > 0 && pos < queue_len; s++) {
                if (!sampled_flag[pos]) {
                    sampled_objects.emplace_back(pos);
                    remaining--;
                }
                pos = in_cache.dq[pos].next;
            }
        }
        last_mode_arm = 0;
    }

    scan_length += (uint64_t)steps;
    if (scan_length >= initial_queue_length) {
        initial_queue_length = (uint32_t)in_cache.metas.size();
        scan_length = 0;
        // Do not soften a committed / injected policy.
        if (arm_phase == ArmPhase::Investigate && !policy_frozen) {
            mab_decay_weights();
        }
    }

    if (!sampled_objects.empty()) {
        spointer_timestamp = in_cache.metas[sampled_objects.back()]._past_timestamp;
        prediction(sampled_objects);
    }

    return (uint32_t)sampled_objects.size();
}

void TLCacheMABCache::evict_with_candidate(pair<uint64_t, uint32_t> &epair) {
    uint64_t key     = epair.first;
    uint32_t old_pos = epair.second;

    if (!mab_off && booster && old_pos < (uint32_t)in_cache.metas.size()) {
        auto it = pred_map.find(key);
        if (it != pred_map.end()) {
            double reward = (double)it->second;

            // Blend short-window miss trend into reward (improving => slight boost).
            if (miss_rate_slope < 0.0) {
                reward *= (1.0 + std::min(0.2, -miss_rate_slope));
            } else if (miss_rate_slope > 0.0) {
                reward *= (1.0 - std::min(0.2, miss_rate_slope));
            }

            uint8_t arm = (arm_strategy == ArmStrategy::MODE)
                              ? last_mode_arm
                              : mab_get_arm(old_pos);

            // TEST is read-only; only TRAIN may update Exp3.
            const bool allow_learn =
                !policy_frozen &&
                profiler.get_mode() != ProfilerMode::TEST;

            if (allow_learn && arm_phase == ArmPhase::Investigate) {
                if (reward > mab_reward_max) {
                    mab_reward_max = reward;
                } else {
                    mab_reward_max = std::max(reward, mab_reward_max * MAB_REWARD_MAX_DECAY);
                }
                mab_update_weight_stable(arm, reward, mab_effective_gamma());
                mab_maybe_commit();
            }
            mab_arm_eviction_count[arm]++;
            mab_total_evictions++;
        }
        if (arm_slider_active()) {
            const int zone = arm_zone_of_pos(old_pos);
            if (zone >= 0) zone_evictions[static_cast<size_t>(zone)]++;
        }
    }

    TLCacheCache::evict_with_candidate(epair);
}

void TLCacheMABCache::mab_compute_probs() {
    double sum_w = 0.0;
    for (int i = 0; i < mab_k; i++) {
        if (mab_weights[i] < 1e-12) mab_weights[i] = 1e-12;
        sum_w += mab_weights[i];
    }
    if (sum_w <= 0.0) sum_w = 1.0;
    const double g = mab_effective_gamma();
    for (int i = 0; i < mab_k; i++) {
        mab_probs[i] = (1.0 - g) * (mab_weights[i] / sum_w) + g / (double)mab_k;
    }
}

void TLCacheMABCache::mab_allocate_samples() {
    uint32_t allocated = 0;
    int      max_arm   = 0;
    for (int i = 0; i < mab_k; i++) {
        mab_arm_samples[i] = (uint32_t)(mab_probs[i] * (double)sample_rate);
        allocated += mab_arm_samples[i];
        if (mab_probs[i] > mab_probs[max_arm]) max_arm = i;
    }
    if (allocated < sample_rate) mab_arm_samples[max_arm] += sample_rate - allocated;
}

uint8_t TLCacheMABCache::mab_get_arm(uint32_t pos) const {
    if (pos >= in_cache.metas.size()) return 0;
    const auto &meta = in_cache.metas[pos];
    switch (arm_strategy) {
    case ArmStrategy::MODE:
        return last_mode_arm;
    case ArmStrategy::POSITION: {
        // Walk-rank approximation: use meta index only for ablation.
        uint32_t queue_len = (uint32_t)in_cache.metas.size();
        if (queue_len <= 1) return 0;
        uint8_t arm = (uint8_t)(((uint64_t)pos * mab_k) / queue_len);
        return arm >= mab_k ? mab_k - 1 : arm;
    }
    case ArmStrategy::AGE: {
        uint64_t age = current_seq - meta._past_timestamp;
        uint64_t max_age = current_seq - in_cache.metas[in_cache.q.head]._past_timestamp;
        if (max_age == 0) return 0;
        uint8_t arm = (uint8_t)((age * (uint64_t)mab_k) / (max_age + 1));
        return arm >= mab_k ? mab_k - 1 : arm;
    }
    case ArmStrategy::FREQUENCY: {
        uint16_t freq = meta._freq;
        if (freq <= 1) return 0;
        int bucket = 0;
        uint16_t f = freq;
        while (f > 1) { f >>= 1; bucket++; }
        return bucket >= (int)mab_k ? mab_k - 1 : (uint8_t)bucket;
    }
    }
    return 0;
}

void TLCacheMABCache::mab_normalize_weights() {
    double max_w = 0.0;
    for (int i = 0; i < mab_k; i++) {
        if (mab_weights[i] > max_w) max_w = mab_weights[i];
    }
    if (max_w <= 0.0) {
        for (int i = 0; i < mab_k; i++) mab_weights[i] = 1.0;
        return;
    }

    const double min_allowed = max_w * MAB_MIN_WEIGHT_RATIO;
    for (int i = 0; i < mab_k; i++) {
        if (mab_weights[i] < min_allowed) mab_weights[i] = min_allowed;
    }

    // Keep mean weight = 1 to avoid float drift / explosion.
    double sum = 0.0;
    for (int i = 0; i < mab_k; i++) sum += mab_weights[i];
    if (sum > 0.0) {
        for (int i = 0; i < mab_k; i++) {
            mab_weights[i] = mab_weights[i] / sum * (double)mab_k;
        }
    }
}

void TLCacheMABCache::mab_update_weight(uint8_t arm, double reward) {
    mab_update_weight_stable(arm, reward, mab_gamma);
}

void TLCacheMABCache::mab_update_weight_stable(uint8_t arm, double reward, double effective_gamma) {
    if (arm >= mab_k) return;
    if (mab_reward_max <= 0.0) mab_reward_max = 1.0;
    if (mab_probs[arm] < 1e-12) mab_compute_probs();
    if (mab_probs[arm] < 1e-12) return;

    double r_norm = reward / mab_reward_max;
    if (r_norm < 0.0) r_norm = 0.0;
    if (r_norm > 1.0) r_norm = 1.0;

    // Pure Exp3 estimator (no aggressiveness_factor).
    double r_hat = r_norm / mab_probs[arm];
    double eta = effective_gamma * r_hat / (double)mab_k;
    if (eta > MAB_ETA_CLIP) eta = MAB_ETA_CLIP;
    if (eta < -MAB_ETA_CLIP) eta = -MAB_ETA_CLIP;

    mab_weights[arm] *= exp(eta);
    mab_normalize_weights();
    mab_compute_probs();
}

void TLCacheMABCache::mab_decay_weights() {
    for (int i = 0; i < mab_k; i++) {
        mab_weights[i] = 0.85 * mab_weights[i] + 0.15;
    }
    mab_normalize_weights();
    mab_compute_probs();
}
