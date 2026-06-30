#include "TLCacheQL.h"

#include <cmath>
#include <iostream>
#include <algorithm>

using namespace std;

namespace TLCacheQL {

// ---------------------------------------------------------------------------
// init_with_params
// ---------------------------------------------------------------------------
void TLCacheQLCache::init_with_params(const map<string, string>& params) {
    // Strip QL-specific params so the base does not warn about them.
    map<string, string> base_params;
    for (auto& kv : params) {
        if (kv.first == "q_alpha") {
            q_alpha = stof(kv.second);
        } else if (kv.first == "q_epsilon") {
            q_epsilon = stof(kv.second);
        } else {
            base_params[kv.first] = kv.second;
        }
    }
    TLCacheCache::init_with_params(base_params);
}

// ---------------------------------------------------------------------------
// encode_state: map (freq, age, size) onto a 9-bit uint32_t
// ---------------------------------------------------------------------------
uint32_t TLCacheQLCache::encode_state(const TLCache::Meta& meta) const {
    uint8_t freq_bucket = (meta._freq == 0) ? 0
        : (uint8_t)min(7, (int)floor(log2((double)meta._freq)));

    uint64_t age = (current_seq > meta._past_timestamp)
        ? (current_seq - meta._past_timestamp) : 0;
    uint32_t ql = (initial_queue_length > 0) ? initial_queue_length : 1;
    uint8_t age_bucket = (uint8_t)min(7, (int)(8 * age / ql));

    uint8_t size_bucket = (meta._size == 0) ? 0
        : (uint8_t)min(7, (int)floor(log2((double)(meta._size / 1024 + 1))));

    return ((uint32_t)freq_bucket << 6) | ((uint32_t)age_bucket << 3) | (uint32_t)size_bucket;
}

// ---------------------------------------------------------------------------
// apply_reward: one-step Q-update for a pending eviction decision
// ---------------------------------------------------------------------------
void TLCacheQLCache::apply_reward(uint64_t key, float reward) {
    auto pit = pending_updates.find(key);
    if (pit == pending_updates.end()) return;

    uint32_t state  = pit->second.state;
    uint8_t  action = pit->second.action;

    auto& qval = q_table[state];     // default-constructs {0,0} if missing
    qval[action] = (1.0f - q_alpha) * qval[action] + q_alpha * reward;
    q_table_updates++;

    pending_updates.erase(pit);
}

// ---------------------------------------------------------------------------
// lookup: intercept re-requests to apply Q-table rewards
// ---------------------------------------------------------------------------
bool TLCacheQLCache::lookup(const SimpleRequest& req) {
    // Pre-check: is req.id currently in out_cache? (→ negative reward if re-requested)
    bool in_out_cache = false;
    {
        auto it = key_map.find(req.id);
        if (it != key_map.end() && it->second.list_idx == 1)
            in_out_cache = true;
    }

    // If req.id is NOT in key_map but has a pending_update, it aged out of
    // out_cache without being re-requested → apply delayed positive reward.
    if (!in_out_cache) {
        auto pit = pending_updates.find(req.id);
        if (pit != pending_updates.end())
            apply_reward(req.id, +1.0f);
    }

    // Also intercept the oldest out_cache entry that is about to be aged out by
    // erase_out_cache() (which runs inside the base lookup, non-virtually).
    if (!out_cache.metas.empty()) {
        uint32_t max_out = (uint32_t)(in_cache.metas.size() * (hsw - 1) + 2);
        if ((uint32_t)out_cache.metas.size() >= max_out) {
            auto& oldest = out_cache.metas[0];
            // oldest._key != req.id: if it equals req.id the negative-reward path handles it
            if (oldest._size != 0 && oldest._key != req.id) {
                auto pit = pending_updates.find(oldest._key);
                if (pit != pending_updates.end())
                    apply_reward(oldest._key, +1.0f);
            }
        }
    }

    bool result = TLCacheCache::lookup(req);

    // Negative reward: object was in out_cache and just got re-requested (cache miss)
    if (in_out_cache) {
        auto pit = pending_updates.find(req.id);
        if (pit != pending_updates.end())
            apply_reward(req.id, -1.0f);
    }

    return result;
}

// ---------------------------------------------------------------------------
// ql_rank: Q-table-driven candidate selection (replaces heuristic rank())
// ---------------------------------------------------------------------------
uint32_t TLCacheQLCache::ql_rank() {
    vector<uint32_t> sampled_objects;

    if (initial_queue_length == 0)
        initial_queue_length = in_cache.metas.size();

    sample_rate = 1024;
    if (sample_rate >= initial_queue_length * 0.01 + eviction_rate)
        sample_rate = initial_queue_length > 2
            ? (uint16_t)(initial_queue_length * 0.01 + eviction_rate) : 2;

    sampled_objects = quick_demotion();

    if (new_obj_size < _currentSize * reserved_space / 10) {
        unsigned int idx_row  = 0;
        int          wraps    = 0;

        while (idx_row < sample_rate &&
               sampled_objects.size() < initial_queue_length) {
            uint32_t pos  = samplepointer;
            auto&    meta = in_cache.metas[pos];
            uint32_t state = encode_state(meta);

            // ε-greedy action selection
            bool nominate;
            if (_distribution(_generator) % 100 <
                    (size_t)(q_epsilon * 100.0f + 0.5f)) {
                // Explore: random action
                nominate = (_distribution(_generator) % 2 == 0);
            } else {
                // Exploit: pick action with higher Q-value
                auto it = q_table.find(state);
                if (it == q_table.end()) {
                    nominate = true;  // unseen state → nominate by default
                } else {
                    nominate = (it->second[1] >= it->second[0]);
                }
            }

            if (nominate) {
                sampled_objects.emplace_back(pos);
                idx_row++;
            }

            scan_length++;
            if (scan_length >= initial_queue_length) {
                // Full-scan wraparound: reset pointers and counters
                initial_queue_length = in_cache.metas.size();
                sample_rate = 1024;
                if (sample_rate >= initial_queue_length * 0.01 + eviction_rate)
                    sample_rate = initial_queue_length > 2
                        ? (uint16_t)(initial_queue_length * 0.01 + eviction_rate)
                        : 2;
                idx_row = 0;
                samplepointer = in_cache.q.head;
                scan_length   = 0;
                pred_map.clear();
                pred_times.clear();
                pred_times.shrink_to_fit();
                wraps++;
                // Guard: stop if we already collected some candidates or spun twice
                if (!sampled_objects.empty() || wraps >= 2)
                    break;
                continue;
            }

            samplepointer = in_cache.dq[samplepointer].next;
        }

        if (!sampled_objects.empty())
            spointer_timestamp =
                in_cache.metas[sampled_objects.back()]._past_timestamp;
        evcition_distribution[1] += sample_rate;
    }

    // Guarantee at least one candidate so evict_predobj never sees an empty heap
    if (sampled_objects.empty())
        sampled_objects.emplace_back(in_cache.q.head);

    prediction(sampled_objects);
    return (uint32_t)sampled_objects.size();
}

// ---------------------------------------------------------------------------
// pop_victim: drain pred_times max-heap to get the highest-reuse-time victim
// ---------------------------------------------------------------------------
pair<uint64_t, uint32_t> TLCacheQLCache::pop_victim() {
    float    reuse_time;
    uint64_t key;
    while (!pred_times.empty()) {
        reuse_time = pred_times.front().reuse_time;
        key        = pred_times.front().key;
        pop_heap(pred_times.begin(), pred_times.end(),
            [](const TLCache::HeapUint& a, const TLCache::HeapUint& b) {
                return a.reuse_time < b.reuse_time;
            });
        pred_times.pop_back();
        if (pred_map.find(key) != pred_map.end() &&
            pred_map[key] == reuse_time) {
            uint32_t old_pos = key_map.find(key)->second.list_pos;
            object_distribution_n_eviction[
                uint16_t(log2(in_cache.metas[old_pos]._freq))]++;
            if (in_cache.metas[old_pos]._past_timestamp <= spointer_timestamp)
                evcition_distribution[0]++;
            return {key, old_pos};
        }
    }
    return {(uint64_t)-1, (uint32_t)-1};
}

// ---------------------------------------------------------------------------
// evict_with_candidate_tracking: record pending update then delegate to base
// ---------------------------------------------------------------------------
void TLCacheQLCache::evict_with_candidate_tracking(
        pair<uint64_t, uint32_t>& epair) {
    if (booster && epair.first != (uint64_t)-1 &&
        epair.second < in_cache.metas.size()) {
        pending_updates[epair.first] = {
            encode_state(in_cache.metas[epair.second]), 1};
    }
    TLCacheCache::evict_with_candidate(epair);
}

// ---------------------------------------------------------------------------
// evict: override virtual evict() to use Q-table rank instead of base rank()
// ---------------------------------------------------------------------------
void TLCacheQLCache::evict() {
    if (!booster) {
        // Cold start: delegate to base (LRU path)
        TLCacheCache::evict();
        return;
    }

    if (evict_nums <= 0 || pred_map.empty()) {
        evict_nums = (int)(ql_rank() / eviction_rate);
    }

    auto epair = pop_victim();
    if (epair.first == (uint64_t)-1) {
        uint32_t pos = in_cache.q.head;
        epair = {in_cache.metas[pos]._key, pos};
    }

    evict_with_candidate_tracking(epair);
}

// ---------------------------------------------------------------------------
// update_stat_periodic: log Q-table activity alongside base stats
// ---------------------------------------------------------------------------
void TLCacheQLCache::update_stat_periodic() {
    TLCacheCache::update_stat_periodic();
    cerr << "q_table_updates=" << q_table_updates
         << " q_table_size=" << q_table.size() << endl;
}

}  // namespace TLCacheQL
