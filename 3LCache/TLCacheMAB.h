#ifndef WEBCACHESIM_TLCACHE_MAB_H
#define WEBCACHESIM_TLCACHE_MAB_H

#include "TLCache.h"
#include <algorithm>
#include <array>
#include <map>
#include <string>
#include <vector>
#include "WorkloadProfiler.hpp"
#include "DiagLogger.hpp"

namespace TLCache {

// MODE arms (default, Path A). Mix is Q1 newest .. Q4 oldest:
//   0 = Balanced ends  (0.50, 0.00, 0.00, 0.50)
//   1 = Tail+          (0.00, 0.00, 0.50, 0.50)
//   2 = Head+          (0.50, 0.50, 0.00, 0.00)
//   3 = Explore        (0.25, 0.25, 0.25, 0.25)
//   4 = Lean middle    (0.15, 0.35, 0.35, 0.15)
//   5 = Lean ends      (0.35, 0.15, 0.15, 0.35)
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

    static const uint8_t MODE_ARM_COUNT = 6;
    uint8_t  mab_k       = MODE_ARM_COUNT;
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

    // KNN picks one start arm at regime open. The parent's LightGBM still
    // ranks objects. The article slider may move band shares inside the
    // regime, but it never changes the regime's saved start arm.
    // Accounting zones for the lap rule: the arc's leading half, its trailing
    // half, and everything outside it.
    static constexpr int N_ZONES = 3;
    static constexpr int ZONE_LEAD = 0;
    static constexpr int ZONE_TRAIL = 1;
    static constexpr int ZONE_OUT = 2;
    // One unit per lap, matching sampling_lru's ++/-- of one percent of the
    // queue. Floor mirrors the original's "never below 1".
    static constexpr double ARC_UNIT = 0.01;
    static constexpr double ARC_MIN_WIDTH = 0.02;
    // Below 1.0 so an adapting arc always leaves an outside to measure.
    static constexpr double ARC_MAX_WIDTH = 0.90;
    // How far the nudge may walk an arc away from the arm that chose it, as
    // an absolute offset and as a relative change in width.
    static constexpr double ARC_MAX_DRIFT = 0.08;
    static constexpr double ARC_WIDTH_SPAN = 0.20;
    static constexpr uint32_t ARM_PROBE_BLOCK_SELECTS = 20;
    static constexpr uint32_t ARM_PROBE_WASH_SELECTS = 5;
    static constexpr uint32_t ARM_PROBE_CYCLES = 2;
    static constexpr uint32_t ARM_PROBE_BLOCKS = 4 * ARM_PROBE_CYCLES;
    static constexpr uint64_t ARM_PROBE_MIN_REQUESTS = 32;
    static constexpr uint64_t ARM_RECHECK_REQUESTS = 2000;
    static constexpr double   ARM_RECHECK_MIN_LOSS = 0.002;
    static constexpr uint32_t ARM_RECHECK_STRIKES = 2;

    struct ArmProbeOutcome {
        uint64_t requests = 0;
        uint64_t bytes = 0;
        uint64_t miss_bytes = 0;
        uint64_t shadow_miss_bytes = 0;
        uint32_t first_order = UINT32_MAX;
    };

    std::array<ArmProbeOutcome, 4> arm_probe_outcomes;
    std::array<uint8_t, ARM_PROBE_BLOCKS> arm_probe_schedule;
    uint32_t arm_probe_block = 0;
    uint32_t arm_probe_selects_in_block = 0;
    bool arm_probe_active = false;
    bool arm_context_ready = false;
    bool arm_model_prediction_ready = false;
    std::vector<double> arm_regime_context;
    int previous_selector_arm = -1;
    double previous_selector_delta = 0.0;

    uint64_t arm_selected_requests = 0;
    uint64_t arm_selected_bytes = 0;
    uint64_t arm_selected_miss_bytes = 0;
    uint64_t arm_selected_shadow_miss_bytes = 0;

    // Rolling chunk used only to judge the currently selected arm.
    uint64_t arm_recheck_requests = 0;
    uint64_t arm_recheck_bytes = 0;
    uint64_t arm_recheck_miss_bytes = 0;
    uint64_t arm_recheck_shadow_miss_bytes = 0;
    uint32_t arm_recheck_strikes = 0;
    // Arms already retired in this regime, so a re-pick makes progress
    // instead of reselecting the arm that just failed. Once all four have
    // failed there is nothing left to switch to, so stop re-checking rather
    // than re-picking the same arm every chunk.
    uint8_t  arm_failed_mask = 0;
    bool     arm_recheck_exhausted = false;
    uint64_t arm_reselect_count = 0;
    uint64_t arm_slider_steps = 0;

    // An arm is one arc of the queue. dq is a CircleList, so the queue is a
    // ring and every arm, including the ones that want both ends, is a single
    // contiguous stretch. Offsets are fractions of the ring measured from
    // q.head, the LRU end, and the arc is [arc_lo, arc_lo + arc_width).
    double arc_lo = 0.0;
    double arc_width = 1.0;
    double start_arc_lo = 0.0;
    double start_arc_width = 1.0;
    std::array<uint64_t, N_ZONES> zone_samples{{0, 0, 0}};
    std::array<uint64_t, N_ZONES> zone_evictions{{0, 0, 0}};

    // Persistent scan cursor, the same idea as TLCacheCache::samplepointer: a
    // call resumes where the last one stopped rather than restarting at
    // q.head, so one call costs a batch of steps instead of a whole lap.
    uint32_t arc_pointer = UINT32_MAX;
    uint32_t arc_scan_pos = 0;
    uint32_t arc_lap_len = 0;
    uint32_t arc_probe_used = 0;

    // Zone of each sampled queue slot, stamped with the current lap.
    // Walking the queue per eviction to recover it is O(queue length), which
    // is fatal on small-object traces where the queue holds tens of thousands
    // of objects.
    std::vector<uint8_t> pos_zone_;
    std::vector<uint32_t> pos_zone_stamp_;
    uint32_t pos_zone_epoch_ = 0;
    int arm_cell_id = -1;
    uint8_t start_arm = 3;
    // Per stream-cell shuffled queue of the MODE start arms (TRAIN collect).
    std::map<int, std::array<uint8_t, MODE_ARM_COUNT>> arm_cell_queue;
    std::map<int, uint8_t> arm_cell_next;

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
    void arm_selector_begin_regime();
    void arm_selector_finish_regime(const char* reason);
    void arm_selector_score_request(uint32_t size, bool is_miss, bool shadow_is_miss);
    void arm_build_context();
    void arm_begin_decision_segment();
    void arm_maybe_redecide();
    void arm_probe_build_schedule();
    void arm_probe_commit();
    bool arm_probe_complete() const;
    bool arm_slider_active() const;
    int arm_stream_cell_id() const;
    uint8_t arm_rotate_start();
    void arm_set_arc_from_arm(uint8_t arm);
    // -1 when this slot was not sampled in the current lap.
    int arm_zone_of_pos(uint32_t pos) const;
    void arm_zone_begin_lap(uint32_t queue_len);
    void arm_zone_remember(uint32_t pos, int zone);
    void arm_sample_arc(std::vector<uint32_t>& sampled_objects, uint32_t& steps_out);
    void arm_arc_lap_end();
    void diag_maybe_snapshot(const char* reason_event = nullptr);
    const char* profiler_mode_cstr() const;
};

} // namespace TLCache

#endif // WEBCACHESIM_TLCACHE_MAB_H
