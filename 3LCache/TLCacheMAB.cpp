#include "TLCacheMAB.h"
#include <cmath>
#include <algorithm>

using namespace std;
using namespace TLCache;

// משתני הטורניר הגלובליים לחלון הזמן הנע
namespace {
    uint64_t global_seq           = 0;
    uint64_t base_window_misses   = 0;
    uint64_t victor_window_misses = 0;
    bool     active_leader_is_mab = false;
}

// ---------------------------------------------------------------------------
// init_with_params
// ---------------------------------------------------------------------------
void TLCacheMABCache::init_with_params(const map<string, string> &params) {
    TLCacheCache::init_with_params(params);

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
        } else if (it.first == "mab_gamma") {
            mab_gamma = stod(it.second);
            if (mab_gamma <= 0.0) mab_gamma = 0.01;
            if (mab_gamma >= 1.0) mab_gamma = 0.99;
        }
    }

    for (int i = 0; i < MAX_ARMS; i++) {
        mab_weights[i]          = 1.0;
        mab_probs[i]            = 1.0 / mab_k;
        mab_arm_samples[i]      = 0;
        mab_arm_eviction_count[i] = 0;
    }
    mab_reward_max    = 1.0;
    mab_total_evictions = 0;

    // הגנה ואיפוס משתני הטורניר בהקצאת המטמון הראשי
    if (!is_shadow_instance) {
        global_seq           = 0;
        base_window_misses   = 0;
        victor_window_misses = 0;
        active_leader_is_mab = false;

        if (shadow_base == nullptr && shadow_victor == nullptr) {
            shadow_base = new TLCacheCache();
            shadow_base->setSize(this->getSize());
            shadow_base->init_with_params(params);

            TLCacheMABCache* sv = new TLCacheMABCache();
            sv->is_shadow_instance = true;
            sv->setSize(this->getSize());
            sv->init_with_params(params);
            shadow_victor = sv;
        }
    }
}

// ---------------------------------------------------------------------------
// lookup
// ---------------------------------------------------------------------------
bool TLCacheMABCache::lookup(const SimpleRequest &req) {
    // אם זו ישות צל, היא פועלת ישירות על תשתית האב ללא ניתוב מחדש
    if (is_shadow_instance) {
        return TLCacheCache::lookup(req);
    }

    // 1. עדכון וניהול עצמאי של יקום הבייסליין המקורי
    if (shadow_base != nullptr) {
        bool base_hit = shadow_base->lookup(req);
        if (!base_hit) {
            base_window_misses++;
            shadow_base->admit(req);
            while (shadow_base->_currentSize > shadow_base->_cacheSize) {
                shadow_base->evict();
            }
        }
    }

    // 2. עדכון וניהול עצמאי של יקום הבנדיט של ויקטור
    if (shadow_victor != nullptr) {
        TLCacheMABCache* sv = static_cast<TLCacheMABCache*>(shadow_victor);
        bool victor_hit = sv->lookup(req);
        if (!victor_hit) {
            victor_window_misses++;
            sv->admit(req);
            while (sv->_currentSize > sv->_cacheSize) {
                sv->evict();
            }
        }
    }

    // 3. קביעת המנצח הדינמי בכל חלון נע (מבוסס מדד פספוסים פיזי ואמיתי)
    global_seq++;
    if (global_seq % 10000 == 0) {
        TLCacheMABCache* sv = static_cast<TLCacheMABCache*>(shadow_victor);
        if (victor_window_misses < base_window_misses) {
            active_leader_is_mab = true;
            sv->mab_gamma = 0.01; // ויקטור מוביל בשטח - עוברים לניצול ממוקד (Exploit)
        } else {
            active_leader_is_mab = false;
            sv->mab_gamma = 0.45; // המאמר מוביל - מאלצים את ויקטור לחקור באגרסיביות בצד
        }
        base_window_misses   = 0;
        victor_window_misses = 0;
    }

    // 4. הרצה על המטמון הראשי (השלישי) שמחזיר תשובות לסימולטור
    return TLCacheCache::lookup(req);
}

// ---------------------------------------------------------------------------
// rank
// ---------------------------------------------------------------------------
uint32_t TLCacheMABCache::rank() {
    // ישות צל של ויקטור תמיד תריץ את ה-MAB. המטמון הראשי יריץ אותו רק אם ויקטור מוביל בטורניר
    if (!is_shadow_instance && !active_leader_is_mab) {
        return TLCacheCache::rank();
    }

    // הגנה מפני מודל לא מאומן או קאש ריק למניעת קריסות
    if (!booster || in_cache.metas.empty()) {
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
    mab_allocate_samples();

    uint32_t arm_collected[MAX_ARMS] = {0};
    uint32_t pos = in_cache.q.head;
    uint32_t steps = 0;
    const uint32_t queue_len = (uint32_t)in_cache.metas.size();
    const uint32_t max_steps = queue_len;

    if (pos >= queue_len) return TLCacheCache::rank();

    sampled_objects.reserve(sample_rate);

    int arms_remaining = 0;
    for (int i = 0; i < mab_k; i++)
        if (mab_arm_samples[i] > 0) arms_remaining++;

    while (steps < max_steps && arms_remaining > 0
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

    scan_length += (uint64_t)steps;
    if (scan_length >= initial_queue_length) {
        initial_queue_length = (uint32_t)in_cache.metas.size();
        scan_length = 0;
        mab_decay_weights();
    }

    if (!sampled_objects.empty()) {
        spointer_timestamp = in_cache.metas[sampled_objects.back()]._past_timestamp;
        prediction(sampled_objects);
    }

    return (uint32_t)sampled_objects.size();
}

// ---------------------------------------------------------------------------
// evict_with_candidate
// ---------------------------------------------------------------------------
void TLCacheMABCache::evict_with_candidate(pair<uint64_t, uint32_t> &epair) {
    uint64_t key     = epair.first;
    uint32_t old_pos = epair.second;

    // עדכון משקלי ה-EXP3 מתבצע בישות הצל של ויקטור, או במטמון הראשי כשהוא מוביל
    if (is_shadow_instance || active_leader_is_mab) {
        if (booster && old_pos < (uint32_t)in_cache.metas.size()) {
            auto it = pred_map.find(key);
            if (it != pred_map.end()) {
                double reward = (double)it->second;
                uint8_t arm = mab_get_arm(old_pos);
                if (reward > mab_reward_max) mab_reward_max = reward;

                mab_update_weight(arm, reward);
                mab_arm_eviction_count[arm]++;
                mab_total_evictions++;
            }
        }
    }

    TLCacheCache::evict_with_candidate(epair);
}

// ---------------------------------------------------------------------------
// EXP3 Helpers
// ---------------------------------------------------------------------------
void TLCacheMABCache::mab_compute_probs() {
    double sum_w = 0.0;
    for (int i = 0; i < mab_k; i++) sum_w += mab_weights[i];
    for (int i = 0; i < mab_k; i++) {
        mab_probs[i] = (1.0 - mab_gamma) * (mab_weights[i] / sum_w) + mab_gamma / (double)mab_k;
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
    case ArmStrategy::POSITION: {
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

void TLCacheMABCache::mab_update_weight(uint8_t arm, double reward) {
    if (mab_reward_max <= 0.0 || arm >= mab_k) return;
    double r_norm = reward / mab_reward_max;
    double r_hat  = r_norm / mab_probs[arm];
    mab_weights[arm] *= exp(mab_gamma * r_hat / (double)mab_k);
    double max_w = 0.0;
    for (int i = 0; i < mab_k; i++) if (mab_weights[i] > max_w) max_w = mab_weights[i];
    if (max_w > 1e6) { for (int i = 0; i < mab_k; i++) mab_weights[i] /= max_w; }
}

void TLCacheMABCache::mab_decay_weights() {
    for (int i = 0; i < mab_k; i++) mab_weights[i] = 0.5 * mab_weights[i] + 0.5;
}