#include "TLCacheMAB.h"
#include <cmath>
#include <algorithm>

using namespace std;
using namespace TLCache;

// ---------------------------------------------------------------------------
// init_with_params
//   Calls parent initialisation first, then parses MAB-specific params.
// ---------------------------------------------------------------------------
void TLCacheMABCache::init_with_params(const map<string, string> &params) {
    // Forward everything to the parent (handles objective, sample_rate, etc.)
    TLCacheCache::init_with_params(params);

    // Parse MAB-specific parameters
    for (auto &it : params) {
        if (it.first == "arm_count") {
            int k = stoi(it.second);
            if (k < 1) k = 1;
            if (k > (int)MAX_ARMS) k = (int)MAX_ARMS;
            mab_k = (uint8_t)k;
        } else if (it.first == "arm_strategy") {
            if (it.second == "position")       arm_strategy = ArmStrategy::POSITION;
            else if (it.second == "frequency") arm_strategy = ArmStrategy::FREQUENCY;
            else if (it.second == "age")       arm_strategy = ArmStrategy::AGE;
            // Unknown strategies silently default to POSITION
        } else if (it.first == "mab_gamma") {
            mab_gamma = stod(it.second);
            if (mab_gamma <= 0.0) mab_gamma = 0.01;
            if (mab_gamma >= 1.0) mab_gamma = 0.99;
        }
    }

    // Initialise EXP3 state
    for (int i = 0; i < MAX_ARMS; i++) {
        mab_weights[i]          = 1.0;
        mab_probs[i]            = 1.0 / mab_k;
        mab_arm_samples[i]      = 0;
        mab_arm_eviction_count[i] = 0;
    }
    mab_reward_max    = 1.0;
    mab_total_evictions = 0;
}

// ---------------------------------------------------------------------------
// mab_compute_probs
//   EXP3: P_i = (1 - gamma) * w_i / sum(w) + gamma / K
// ---------------------------------------------------------------------------
void TLCacheMABCache::mab_compute_probs() {
    double sum_w = 0.0;
    for (int i = 0; i < mab_k; i++) sum_w += mab_weights[i];

    for (int i = 0; i < mab_k; i++) {
        mab_probs[i] = (1.0 - mab_gamma) * (mab_weights[i] / sum_w)
                       + mab_gamma / (double)mab_k;
    }
}

// ---------------------------------------------------------------------------
// mab_allocate_samples
//   Translate probabilities into integer sample counts summing to sample_rate.
//   Remainder is added to the highest-probability arm.
// ---------------------------------------------------------------------------
void TLCacheMABCache::mab_allocate_samples() {
    uint32_t allocated = 0;
    int      max_arm   = 0;
    for (int i = 0; i < mab_k; i++) {
        mab_arm_samples[i] = (uint32_t)(mab_probs[i] * (double)sample_rate);
        allocated += mab_arm_samples[i];
        if (mab_probs[i] > mab_probs[max_arm]) max_arm = i;
    }
    // Give remainder to the heaviest arm
    if (allocated < sample_rate)
        mab_arm_samples[max_arm] += sample_rate - allocated;
}

// ---------------------------------------------------------------------------
// mab_get_arm
//   Maps a position in in_cache.metas to arm index [0, mab_k).
//   Supports three strategies:
//     POSITION  – ordinal rank in LRU queue (how far from the tail the
//                 object sits), approximated by walking position relative
//                 to queue size. Oldest (head) → arm 0.
//     FREQUENCY – log2 bucket based on _freq, generalised to K arms.
//     AGE       – time-since-last-access bucketed into K equal-width bins
//                 relative to the oldest object's age.
// ---------------------------------------------------------------------------
uint8_t TLCacheMABCache::mab_get_arm(uint32_t pos) const {
    const auto &meta = in_cache.metas[pos];

    switch (arm_strategy) {

    case ArmStrategy::POSITION: {
        // Use ordinal rank: walk from head to find this object's approximate
        // position.  To avoid O(n) per call, we use the object's index in
        // in_cache.metas as a proxy for insertion order (newer = higher index).
        // arm 0 → oldest (lowest insertion index), arm K-1 → newest.
        uint32_t queue_len = (uint32_t)in_cache.metas.size();
        if (queue_len <= 1) return 0;
        // pos is the index in metas; normalise by queue length.
        uint8_t arm = (uint8_t)(((uint64_t)pos * mab_k) / queue_len);
        if (arm >= mab_k) arm = mab_k - 1;
        return arm;
    }

    case ArmStrategy::AGE: {
        // Age = how long since this object was last touched.
        // Objects at the LRU head have the largest age.
        uint64_t age = current_seq - meta._past_timestamp;
        uint64_t max_age = current_seq
                         - in_cache.metas[in_cache.q.head]._past_timestamp;
        if (max_age == 0) return 0;
        // arm 0 → oldest (age ≈ max_age), arm K-1 → newest (age ≈ 0)
        uint8_t arm = (uint8_t)((age * (uint64_t)mab_k) / (max_age + 1));
        if (arm >= mab_k) arm = mab_k - 1;
        return arm;
    }

    case ArmStrategy::FREQUENCY: {
        // Log2 bucketing generalised to exactly K arms.
        // freq=0 or 1 → arm 0; freq=2 → arm 1; freq=2^(i-1)+1..2^i → arm i;
        // anything beyond arm K-2 maps to arm K-1.
        uint16_t freq = meta._freq;
        if (freq <= 1) return 0;
        // floor(log2(freq)) gives the bucket; clamp to [0, K-1]
        int bucket = 0;
        uint16_t f = freq;
        while (f > 1) { f >>= 1; bucket++; }
        if (bucket >= (int)mab_k) bucket = (int)mab_k - 1;
        return (uint8_t)bucket;
    }
    }
    return 0; // unreachable
}

// ---------------------------------------------------------------------------
// mab_update_weight
//   EXP3 multiplicative update:
//     w_i ← w_i * exp( gamma * r̂ / K )
//   where r̂ = reward / reward_max / P_i   (importance-weighted, normalised)
// ---------------------------------------------------------------------------
void TLCacheMABCache::mab_update_weight(uint8_t arm, double reward) {
    if (mab_reward_max <= 0.0) return;
    double r_norm = reward / mab_reward_max;             // ∈ [0, 1]
    double r_hat  = r_norm / mab_probs[arm];             // importance correction
    mab_weights[arm] *= exp(mab_gamma * r_hat / (double)mab_k);

    // Normalise to prevent floating-point overflow
    double max_w = 0.0;
    for (int i = 0; i < mab_k; i++)
        if (mab_weights[i] > max_w) max_w = mab_weights[i];
    if (max_w > 1e6) {
        for (int i = 0; i < mab_k; i++) mab_weights[i] /= max_w;
    }
}

// ---------------------------------------------------------------------------
// mab_decay_weights
//   Soft-reset toward 1.0 to allow re-exploration after workload shifts.
//   Called once per full scan cycle.
// ---------------------------------------------------------------------------
void TLCacheMABCache::mab_decay_weights() {
    for (int i = 0; i < mab_k; i++) {
        mab_weights[i] = 0.5 * mab_weights[i] + 0.5;
    }
}

// ---------------------------------------------------------------------------
// rank  (overrides TLCacheCache::rank)
//
//   Replaces the original heuristic LRU-scan + quick_demotion sampling with
//   an EXP3 bandit that allocates the sample_rate budget across K arms.
//
//   Each arm corresponds to a region of the cache (defined by arm_strategy).
//   The algorithm:
//     1. Compute EXP3 probabilities from current weights.
//     2. Allocate sample_rate slots across arms proportionally.
//     3. Walk the full LRU list once and collect the required number of
//        objects per arm. (Single pass; O(n) worst-case but usually short.)
//     4. If some arms under-fill (not enough objects in that bucket), do a
//        second pass to fill any remaining budget from anywhere.
//     5. Feed all candidates to prediction() (LightGBM scoring) — unchanged.
// ---------------------------------------------------------------------------
uint32_t TLCacheMABCache::rank() {
    vector<uint32_t> sampled_objects;

    if (initial_queue_length == 0)
        initial_queue_length = in_cache.metas.size();

    // Compute sample_rate (same adaptive logic as parent)
    sample_rate = 1024;
    if (sample_rate >= initial_queue_length * 0.01 + eviction_rate)
        sample_rate = initial_queue_length > 2
                    ? (uint16_t)(initial_queue_length * 0.01 + eviction_rate)
                    : 2;

    // Edge case: empty or near-empty cache
    if (in_cache.metas.empty()) {
        return 0;
    }

    // ---- EXP3 allocation ----
    mab_compute_probs();
    mab_allocate_samples();

    // ---- First pass: collect per-arm quotas ----
    uint32_t arm_collected[MAX_ARMS] = {0};
    uint32_t pos = in_cache.q.head;
    uint32_t steps = 0;
    const uint32_t queue_len = (uint32_t)in_cache.metas.size();
    const uint32_t max_steps = queue_len; // at most one full traversal

    sampled_objects.reserve(sample_rate);

    // Count how many arms still need samples (avoid O(K) check per step).
    int arms_remaining = 0;
    for (int i = 0; i < mab_k; i++)
        if (mab_arm_samples[i] > 0) arms_remaining++;

    while (steps < max_steps && arms_remaining > 0
           && (uint32_t)sampled_objects.size() < sample_rate) {
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

    // ---- Second pass: fill any remaining budget (under-filled arms) ----
    uint32_t remaining = sample_rate - (uint32_t)sampled_objects.size();
    if (remaining > 0 && queue_len > (uint32_t)sampled_objects.size()) {
        // Use a boolean array (O(max_pos)) to mark already-sampled positions.
        // This avoids hash-set overhead when under-fill is rare.
        vector<bool> sampled_flag(queue_len, false);
        for (uint32_t p : sampled_objects) sampled_flag[p] = true;

        pos = in_cache.q.head;
        for (uint32_t s = 0; s < queue_len && remaining > 0; s++) {
            if (pos < queue_len && !sampled_flag[pos]) {
                sampled_objects.emplace_back(pos);
                remaining--;
            }
            pos = in_cache.dq[pos].next;
        }
    }

    // ---- Periodic: scan-cycle bookkeeping ----
    // Track LRU positions walked (not objects sampled) to match the parent's
    // cycle definition — one cycle = one full pass over initial_queue_length.
    scan_length += (uint64_t)steps;
    if (scan_length >= initial_queue_length) {
        initial_queue_length = (uint32_t)in_cache.metas.size();
        scan_length = 0;
        // Decay weights to allow re-exploration
        mab_decay_weights();
        // Recompute sample_rate for next cycle
        sample_rate = 1024;
        if (sample_rate >= initial_queue_length * 0.01 + eviction_rate)
            sample_rate = initial_queue_length > 2
                        ? (uint16_t)(initial_queue_length * 0.01 + eviction_rate)
                        : 2;
    }

    // ---- Run LightGBM prediction on all sampled candidates ----
    if (!sampled_objects.empty()) {
        spointer_timestamp = in_cache.metas[sampled_objects.back()]._past_timestamp;
        prediction(sampled_objects);
    }

    return (uint32_t)sampled_objects.size();
}

// ---------------------------------------------------------------------------
// evict_with_candidate  (overrides TLCacheCache::evict_with_candidate)
//
//   Injects the EXP3 reward update BEFORE delegating to the parent eviction.
//   Reward = predicted reuse time of the evicted object (higher is better:
//   we evicted something that won't be needed soon).
// ---------------------------------------------------------------------------
void TLCacheMABCache::evict_with_candidate(pair<uint64_t, uint32_t> &epair) {
    uint64_t key     = epair.first;
    uint32_t old_pos = epair.second;

    // Update bandit only once the model is trained and this object was a
    // predicted candidate (i.e. it appeared in pred_map from a prior rank()).
    // Skip the update when pred_map has no entry — it means this eviction
    // fell back to LRU and carries no useful signal for the bandit.
    if (booster && old_pos < (uint32_t)in_cache.metas.size()) {
        auto it = pred_map.find(key);
        if (it != pred_map.end()) {
            uint8_t arm = mab_get_arm(old_pos);
            double reward = (double)it->second;

            if (reward > mab_reward_max) mab_reward_max = reward;

            mab_update_weight(arm, reward);
            mab_arm_eviction_count[arm]++;
            mab_total_evictions++;
        }
    }

    // Delegate actual eviction to parent
    TLCacheCache::evict_with_candidate(epair);
}
