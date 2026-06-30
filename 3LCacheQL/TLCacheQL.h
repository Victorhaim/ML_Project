#ifndef WEBCACHESIM_TLCacheQL_H
#define WEBCACHESIM_TLCacheQL_H

#include "../3LCache/TLCache.h"
#include <unordered_map>
#include <array>
#include <map>
#include <string>
#include <utility>
#include <cstdint>

namespace TLCacheQL {

struct PendingUpdate {
    uint32_t state;
    uint8_t  action;  // 0 = skip, 1 = nominate (always 1 for nominated candidates)
};

class TLCacheQLCache : public TLCache::TLCacheCache {
public:
    // Q-table: state (9-bit encoded) -> [Q(s,skip), Q(s,nominate)]
    std::unordered_map<uint32_t, std::array<float, 2>> q_table;
    // Pending reward tracking: evicted key -> {state at eviction, action taken}
    std::unordered_map<uint64_t, PendingUpdate> pending_updates;

    float    q_alpha        = 0.1f;
    float    q_epsilon      = 0.1f;
    uint64_t q_table_updates = 0;

    void init_with_params(const std::map<std::string, std::string>& params) override;
    bool lookup(const SimpleRequest& req) override;
    void evict();
    void update_stat_periodic() override;

    // Public: called from interface file
    uint32_t ql_rank();
    std::pair<uint64_t, uint32_t> pop_victim();
    void evict_with_candidate_tracking(std::pair<uint64_t, uint32_t>& epair);
    uint32_t encode_state(const TLCache::Meta& meta) const;

private:
    void apply_reward(uint64_t key, float reward);
};

}  // namespace TLCacheQL

#endif  // WEBCACHESIM_TLCacheQL_H
