#ifndef WEBCACHESIM_TLCACHE_MAB_H
#define WEBCACHESIM_TLCACHE_MAB_H

#include "TLCache.h"
#include <algorithm>

namespace TLCache {

enum class ArmStrategy : uint8_t {
    POSITION  = 0,
    FREQUENCY = 1,
    AGE       = 2
};

class TLCacheMABCache : public TLCacheCache {
public:
    // --- EXP3 configuration ---
    static const uint8_t MAX_ARMS = 16;
    uint8_t  mab_k       = 4;
    ArmStrategy arm_strategy = ArmStrategy::POSITION;
    double   mab_gamma   = 0.1;

    double   mab_weights[MAX_ARMS];
    double   mab_probs[MAX_ARMS];
    uint32_t mab_arm_samples[MAX_ARMS];
    double   mab_reward_max = 1.0;
    uint64_t mab_total_evictions = 0;
    uint64_t mab_arm_eviction_count[MAX_ARMS];

    // דריסת פונקציות הליבה לניהול שלושת המטמונים במקביל
    bool lookup(const SimpleRequest &req) override;
    uint32_t rank() override;
    void evict_with_candidate(pair<uint64_t, uint32_t> &epair) override;
    void init_with_params(const map<string, string> &params) override;

    // משתני שליטה לבידוד הזיכרון של היקומים המקבילים
    TLCacheCache* shadow_base = nullptr;
    void*         shadow_victor = nullptr; 
    bool          is_shadow_instance = false;

    // Destructor לניקוי זיכרון בטוח בסיום הריצה
    virtual ~TLCacheMABCache() {
        if (!is_shadow_instance) {
            delete shadow_base;
            delete static_cast<TLCacheMABCache*>(shadow_victor);
        }
    }

    void mab_compute_probs();
    void mab_allocate_samples();
    uint8_t mab_get_arm(uint32_t pos) const;
    void mab_update_weight(uint8_t arm, double reward);
    void mab_decay_weights();
};

} // namespace TLCache

#endif // WEBCACHESIM_TLCACHE_MAB_H