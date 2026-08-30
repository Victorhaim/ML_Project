#ifndef WEBCACHESIM_TLCACHE_MAB_H
#define WEBCACHESIM_TLCACHE_MAB_H

#include "TLCache.h"
#include <algorithm>
#include <string>
#include "WorkloadProfiler.hpp"
#include "DiagLogger.hpp"

namespace TLCache {

// MODE arms (default, Path A):
//   0 = Balanced (head+tail mix)
//   1 = Tail+    (favor aged / unpopular side)
//   2 = Head+    (favor newly admitted side)
//   3 = Explore  (wider scan)
// POSITION / FREQUENCY / AGE kept for ablation only.
enum class ArmStrategy : uint8_t {
    MODE      = 0,
    POSITION  = 1,
    FREQUENCY = 2,
    AGE       = 3
};

class TLCacheMABCache : public TLCacheCache {
public:
    static const uint8_t MAX_ARMS = 16;
    static constexpr double MAB_ETA_CLIP = 2.0;
    static constexpr double MAB_MIN_WEIGHT_RATIO = 1e-3;
    static constexpr double MAB_REWARD_MAX_DECAY = 0.999;

    WorkloadProfiler profiler;
    DiagLogger diag;

    // Frozen baseline: parent 3L-Cache eviction, no MAB update, no policy
    // injection. Used to produce a window-aligned reference run.
    bool mab_off = false;

    uint8_t  mab_k       = 4;
    ArmStrategy arm_strategy = ArmStrategy::MODE;
    double   mab_gamma   = 0.05;

    double   mab_weights[MAX_ARMS];
    double   mab_probs[MAX_ARMS];
    uint32_t mab_arm_samples[MAX_ARMS];
    double   mab_reward_max = 1.0;
    uint64_t mab_total_evictions = 0;
    uint64_t mab_arm_eviction_count[MAX_ARMS];
    uint64_t mab_arm_select_count[MAX_ARMS];

    // Arm selected for the latest rank() call (MODE credit assignment).
    uint8_t  last_mode_arm = 0;
    uint64_t mab_rng_state = 0xC0FFEEULL;

    // Feature regime arm clock (independent of feature ε):
    // Investigate (fair pulls / high γ) until ranking stabilizes, then Commit.
    enum class ArmPhase : uint8_t { Investigate = 0, Committed = 1 };
    ArmPhase arm_phase = ArmPhase::Investigate;
    uint64_t arm_regime_selects[MAX_ARMS];
    uint64_t arm_gap_hold = 0;
    uint8_t  committed_arm = 0;
    uint64_t rr_probe_idx = 0;
    // TEST: freeze Exp3 while a table policy is actively injected; the row's
    // own arm is replayed verbatim without learning.
    bool     policy_frozen = false;

    static constexpr uint64_t ARM_N_MIN = 40;      // min selects per arm before commit
    static constexpr double   ARM_GAP = 0.20;      // best vs 2nd weight gap
    static constexpr uint64_t ARM_GAP_HOLD = 5;    // consecutive stabilize checks
    static constexpr double   INVESTIGATE_GAMMA = 0.35;
    static constexpr double   COMMIT_GAMMA = 0.01;

    double   ema_miss_rate     = 0.0;
    double   miss_rate_slope   = 0.0;
    double   last_window_miss  = 0.0;

    uint64_t total_req_count   = 0;
    uint64_t total_miss_count  = 0;

    bool     diag_enable = false;
    uint64_t diag_interval = 10000;
    std::string diag_prefix = "mab_diag";

    bool     train_log_enable = true;
    uint64_t train_log_interval = 10000;
    std::string train_log_prefix = "mab_train";

    uint64_t regime_id      = 0;
    uint64_t inject_count   = 0;
    uint64_t total_bytes_req  = 0;
    uint64_t total_bytes_miss = 0;
    uint64_t total_shadow_misses = 0;
    uint64_t total_shadow_bytes_miss = 0;

    double current_req_clock_time = 0.0;
    bool   current_req_is_write   = false;
    bool   current_shadow_hit     = false;
    void set_request_info(double ct, bool iw, bool shadow_hit) {
        current_req_clock_time = ct;
        current_req_is_write   = iw;
        current_shadow_hit = shadow_hit;
    }
    uint32_t window_req_count  = 0;
    uint32_t window_miss_count = 0;

    uint64_t global_seq = 0;

    std::string policy_file_path = "meta_policy_v3.txt";
    std::string config_file_path = "profiler_config_v3.txt";

    bool lookup(const SimpleRequest &req) override;
    uint32_t rank() override;
    void evict_with_candidate(pair<uint64_t, uint32_t> &epair) override;
    void init_with_params(const map<string, string> &params) override;

    virtual ~TLCacheMABCache();

    void mab_compute_probs();
    void mab_allocate_samples();
    uint8_t mab_get_arm(uint32_t pos) const;
    uint8_t mab_select_arm();
    double mab_next_rand();
    void mab_sample_mode(uint8_t mode, std::vector<uint32_t> &sampled_objects,
                         uint32_t &steps_out);
    void mab_normalize_weights();
    void mab_update_weight(uint8_t arm, double reward);
    void mab_update_weight_stable(uint8_t arm, double reward, double effective_gamma);
    void mab_decay_weights();
    void mab_on_regime_reset();
    void mab_maybe_commit();
    double mab_effective_gamma() const;
    void diag_maybe_snapshot(const char* reason_event = nullptr);
    const char* profiler_mode_cstr() const;
};

} // namespace TLCache

#endif // WEBCACHESIM_TLCACHE_MAB_H
