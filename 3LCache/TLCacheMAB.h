#ifndef WEBCACHESIM_TLCACHE_MAB_H
#define WEBCACHESIM_TLCACHE_MAB_H

#include "TLCache.h"
#include <algorithm>

namespace TLCache {

// -----------------------------------------------------------------------
// Arm strategy: how to assign a cache object to an arm
// -----------------------------------------------------------------------
enum class ArmStrategy : uint8_t {
    POSITION  = 0, // Split by ordinal position in the LRU queue (recency)
    FREQUENCY = 1, // Split by object access frequency (_freq) buckets
    AGE       = 2  // Split by time-since-last-access (timestamp age)
};

// -----------------------------------------------------------------------
// TLCacheMABCache
//   Inherits all lookup / admit / train / predict logic from TLCacheCache.
//   Overrides rank() to use EXP3 multi-armed bandit for candidate sampling,
//   and overrides evict_with_candidate() to inject the reward update.
// -----------------------------------------------------------------------
class TLCacheMABCache : public TLCacheCache {
public:
    // --- Configuration (set via init_with_params) ---
    static const uint8_t MAX_ARMS = 16; // hard upper bound
    uint8_t  mab_k       = 4;           // number of arms (4 or 8)
    ArmStrategy arm_strategy = ArmStrategy::POSITION;
    double   mab_gamma   = 0.1;         // EXP3 exploration parameter

    // --- EXP3 state ---
    double   mab_weights[MAX_ARMS];     // per-arm weights (init = 1.0)
    double   mab_probs[MAX_ARMS];       // computed selection probabilities
    uint32_t mab_arm_samples[MAX_ARMS]; // samples to collect per arm each round

    // Reward normalisation: track running max predicted reuse time
    double   mab_reward_max = 1.0;

    // Per-arm eviction stats (debug / analysis)
    uint64_t mab_total_evictions = 0;
    uint64_t mab_arm_eviction_count[MAX_ARMS];

    // ---------------------------------------------------------------
    // Overridden public interface
    // ---------------------------------------------------------------
    void init_with_params(const map<string, string> &params) override;
    uint32_t rank() override;
    void evict_with_candidate(pair<uint64_t, uint32_t> &epair) override;

    // ---------------------------------------------------------------
    // MAB helpers (public so they can be unit-tested if desired)
    // ---------------------------------------------------------------

    // Compute EXP3 probabilities from current weights
    void mab_compute_probs();

    // Distribute sample_rate budget across arms proportionally
    void mab_allocate_samples();

    // Map a position in in_cache.metas to an arm index [0, mab_k)
    uint8_t mab_get_arm(uint32_t pos) const;

    // EXP3 weight update after an eviction
    void mab_update_weight(uint8_t arm, double reward);

    // Decay all weights toward 1.0 at end of each scan cycle
    void mab_decay_weights();
};

} // namespace TLCache

#endif // WEBCACHESIM_TLCACHE_MAB_H
