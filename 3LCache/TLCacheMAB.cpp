#include "TLCacheMAB.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <sstream>
#include <iostream>

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
    double test_match_max_dist = 3.5;

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
        } else if (it.first == "test_match_max_dist" ||
                   it.first == "test-match-max-dist") {
            test_match_max_dist = stod(it.second);
        }
    }

    // Mode strategy is defined for 4 sampling modes.
    if (arm_strategy == ArmStrategy::MODE && mab_k != 4) {
        mab_k = 4;
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
    mab_compute_probs();

    string mode_str = "train";
    auto mode_it = get_param("profiler_mode", "profiler-mode");
    if (mode_it != params.end()) {
        mode_str = mode_it->second;
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
                  train_log_enable, train_log_prefix, train_log_interval);
    profiler.set_test_match_max_dist(test_match_max_dist);
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
    if (profiler.get_mode() == ProfilerMode::TEST && !policy_frozen) {
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

    if (arm_strategy == ArmStrategy::MODE) {
        // Hard arm selection: one sampling mode per rank() for clean Exp3 credit.
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
