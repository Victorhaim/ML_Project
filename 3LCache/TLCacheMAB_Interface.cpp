#include <assert.h>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>

#include "../libCacheSim/dataStructure/hashtable/hashtable.h"
#include "../libCacheSim/include/libCacheSim/cache.h"
#include "TLCacheMAB.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void            *TLCacheMAB_cache;
    // Exact article baseline, advanced once for every request in the same
    // process. Its outcome is the regime win signal; no offline confirm run.
    // Held as the plain parent class so no C wrapper symbol is needed.
    void            *shadow_3l;
    SimpleRequest    shadow_req;
    char            *objective;
    int              arm_count;
    char            *arm_strategy;
    double           mab_gamma;
    char            *profiler_mode;
    char            *policy_file;
    char            *config_file;
    int              diag_enable;   // -1 = auto, 0 = off, 1 = on
    uint64_t         diag_interval;
    char            *diag_prefix;
    int              train_log_enable; // -1 = auto, 0 = off, 1 = on
    uint64_t         train_log_interval;
    char            *train_log_prefix;
    int              mab_off;
    SimpleRequest    TLCacheMAB_req;
    pair<uint64_t, uint32_t> to_evict_pair;
    cache_obj_t      obj_tmp;
} TLCacheMAB_params_t;

static const char *DEFAULT_PARAMS =
    "objective=byte-miss-ratio,arm_count=4,arm_strategy=mode,mab_gamma=0.05,"
    "profiler_mode=train,policy_file=meta_policy_v3.txt,"
    "config_file=profiler_config_v3.txt,"
    "diag_interval=10000,diag_prefix=mab_diag";

static void    TLCacheMAB_free(cache_t *cache);
static bool    TLCacheMAB_get(cache_t *cache, const request_t *req);
static cache_obj_t *TLCacheMAB_find(cache_t *cache, const request_t *req, const bool update_cache);
static cache_obj_t *TLCacheMAB_insert(cache_t *cache, const request_t *req);
static cache_obj_t *TLCacheMAB_to_evict(cache_t *cache, const request_t *req);
static void    TLCacheMAB_evict(cache_t *cache, const request_t *req);
static bool    TLCacheMAB_remove(cache_t *cache, const obj_id_t obj_id);
static int64_t TLCacheMAB_get_occupied_byte(const cache_t *cache);
static int64_t TLCacheMAB_get_n_obj(const cache_t *cache);
static void    TLCacheMAB_parse_params(cache_t *cache, const char *cache_specific_params);

cache_t *TLCacheMAB_init(const common_cache_params_t ccache_params,
                          const char *cache_specific_params) {
    cache_t *cache = cache_struct_init("TLCacheMAB", ccache_params, cache_specific_params);
    cache->cache_init          = TLCacheMAB_init;
    cache->cache_free          = TLCacheMAB_free;
    cache->get                 = TLCacheMAB_get;
    cache->find                = TLCacheMAB_find;
    cache->insert              = TLCacheMAB_insert;
    cache->evict               = TLCacheMAB_evict;
    cache->to_evict            = TLCacheMAB_to_evict;
    cache->remove              = TLCacheMAB_remove;
    cache->can_insert          = cache_can_insert_default;
    cache->get_occupied_byte   = TLCacheMAB_get_occupied_byte;
    cache->get_n_obj           = TLCacheMAB_get_n_obj;
    cache->to_evict_candidate  = static_cast<cache_obj_t *>(malloc(sizeof(cache_obj_t)));

    if (ccache_params.consider_obj_metadata) {
        cache->obj_md_size = 180;
    } else {
        cache->obj_md_size = 0;
    }

    auto *params = my_malloc(TLCacheMAB_params_t);
    memset(params, 0, sizeof(TLCacheMAB_params_t));
    cache->eviction_params = params;

    params->objective     = strdup("byte-miss-ratio");
    params->arm_count     = 4;
    params->arm_strategy  = strdup("mode");
    params->mab_gamma     = 0.05;
    params->profiler_mode = strdup("train");
    params->policy_file   = strdup("meta_policy_v3.txt");
    params->config_file   = strdup("profiler_config_v3.txt");
    params->diag_enable   = -1; // auto: on for test, off for train unless set
    params->diag_interval = 10000;
    params->diag_prefix   = strdup("mab_diag");
    params->train_log_enable = -1; // auto: on for train
    params->train_log_interval = 10000;
    params->train_log_prefix = strdup("mab_train");
    params->mab_off      = 0;

    if (cache_specific_params != NULL) {
        TLCacheMAB_parse_params(cache, cache_specific_params);
    } else {
        TLCacheMAB_parse_params(cache, DEFAULT_PARAMS);
    }

    auto *mab = new TLCache::TLCacheMABCache();
    params->TLCacheMAB_cache = static_cast<void *>(mab);
    mab->setSize(ccache_params.cache_size);

    if (!params->mab_off) {
        auto *shadow = new TLCache::TLCacheCache();
        shadow->setSize(ccache_params.cache_size);
        std::map<std::string, std::string> shadow_map;
        shadow_map["objective"] = params->objective;
        shadow->init_with_params(shadow_map);
        params->shadow_3l = static_cast<void *>(shadow);
    }

    std::map<std::string, std::string> params_map;
    params_map["objective"]      = params->objective;
    params_map["arm_count"]      = std::to_string(params->arm_count);
    params_map["arm_strategy"]   = params->arm_strategy;
    params_map["mab_gamma"]      = std::to_string(params->mab_gamma);
    params_map["profiler_mode"]  = params->profiler_mode;
    params_map["policy_file"]    = params->policy_file;
    params_map["config_file"]    = params->config_file;
    params_map["diag_interval"]  = std::to_string(params->diag_interval);
    params_map["diag_prefix"]    = params->diag_prefix;
    if (params->diag_enable == 0) params_map["diag_enable"] = "0";
    else if (params->diag_enable == 1) params_map["diag_enable"] = "1";
    params_map["train_log_interval"] = std::to_string(params->train_log_interval);
    params_map["train_log_prefix"] = params->train_log_prefix;
    if (params->train_log_enable == 0) params_map["train_log_enable"] = "0";
    else if (params->train_log_enable == 1) params_map["train_log_enable"] = "1";
    params_map["mab_off"]    = params->mab_off ? "1" : "0";
    // diag/train_log enable == -1 => leave unset for C++ auto mode

    mab->init_with_params(params_map);

    if (strcmp(params->objective, "object-miss-ratio") == 0) {
        snprintf(cache->cache_name, CACHE_NAME_ARRAY_LEN,
                 "TLCache-MAB-%s-K%d-OMR",
                 params->arm_strategy, params->arm_count);
    } else {
        snprintf(cache->cache_name, CACHE_NAME_ARRAY_LEN,
                 "TLCache-MAB-%s-K%d-BMR",
                 params->arm_strategy, params->arm_count);
    }

    return cache;
}

static void TLCacheMAB_free(cache_t *cache) {
    auto *params = static_cast<TLCacheMAB_params_t *>(cache->eviction_params);
    auto *mab    = static_cast<TLCache::TLCacheMABCache *>(params->TLCacheMAB_cache);
    if (params->shadow_3l != nullptr) {
        delete static_cast<TLCache::TLCacheCache *>(params->shadow_3l);
        params->shadow_3l = nullptr;
    }
    delete mab;
    free(params->objective);
    free(params->arm_strategy);
    free(params->profiler_mode);
    free(params->policy_file);
    free(params->config_file);
    free(params->diag_prefix);
    free(params->train_log_prefix);
    free(cache->to_evict_candidate);
    my_free(sizeof(TLCacheMAB_params_t), params);
    cache_struct_free(cache);
}

static bool TLCacheMAB_get(cache_t *cache, const request_t *req) {
    return cache_get_base(cache, req);
}

// Mirrors cache_get_base for the shadow: lookup, then evict until the object
// fits and admit. Keeps the shadow byte-for-byte aligned with a standalone
// 3L-Cache run over the same trace.
static bool TLCacheMAB_shadow_step(TLCacheMAB_params_t *params,
                                   const cache_t *cache, const request_t *req) {
    auto *shadow = static_cast<TLCache::TLCacheCache *>(params->shadow_3l);
    params->shadow_req.reinit(cache->n_req, req->obj_id, req->obj_size, nullptr);

    if (shadow->lookup(params->shadow_req)) return true;

    const uint64_t need = (uint64_t)req->obj_size + (uint64_t)cache->obj_md_size;
    if (need <= shadow->getSize()) {
        while (shadow->getCurrentSize() + need > shadow->getSize()) {
            uint64_t before = shadow->getCurrentSize();
            shadow->evict();
            if (shadow->getCurrentSize() >= before) break;
        }
        shadow->admit(params->shadow_req);
    }
    return false;
}

static cache_obj_t *TLCacheMAB_find(cache_t *cache, const request_t *req, const bool update_cache) {
    auto *params = static_cast<TLCacheMAB_params_t *>(cache->eviction_params);
    auto *mab    = static_cast<TLCache::TLCacheMABCache *>(params->TLCacheMAB_cache);

    if (!update_cache) {
        bool is_hit = mab->exist(static_cast<int64_t>(req->obj_id));
        return is_hit ? reinterpret_cast<cache_obj_t *>(0x1) : NULL;
    }

    bool shadow_hit = false;
    if (params->shadow_3l != nullptr) {
        shadow_hit = TLCacheMAB_shadow_step(params, cache, req);
    }

    params->TLCacheMAB_req.reinit(cache->n_req, req->obj_id, req->obj_size, nullptr);
    mab->set_request_info((double)req->clock_time,
                          (req->op == OP_WRITE || req->op == 1),
                          shadow_hit);

    bool is_hit = mab->lookup(params->TLCacheMAB_req);
    return is_hit ? reinterpret_cast<cache_obj_t *>(0x1) : NULL;
}

static cache_obj_t *TLCacheMAB_insert(cache_t *cache, const request_t *req) {
    auto *params = static_cast<TLCacheMAB_params_t *>(cache->eviction_params);
    auto *mab    = static_cast<TLCache::TLCacheMABCache *>(params->TLCacheMAB_cache);
    params->TLCacheMAB_req.reinit(cache->n_req, req->obj_id, req->obj_size, nullptr);

    mab->admit(params->TLCacheMAB_req);
    return reinterpret_cast<cache_obj_t *>(0x1);
}

static cache_obj_t *TLCacheMAB_to_evict(cache_t *cache, const request_t *req) {
    auto *params = static_cast<TLCacheMAB_params_t *>(cache->eviction_params);
    auto *mab    = static_cast<TLCache::TLCacheMABCache *>(params->TLCacheMAB_cache);

    params->to_evict_pair = mab->evict_predobj();
    auto &meta = mab->in_cache.metas[params->to_evict_pair.second];

    params->obj_tmp.obj_id   = params->to_evict_pair.first;
    params->obj_tmp.obj_size = meta._size;

    cache->to_evict_candidate = &params->obj_tmp;
    cache->to_evict_candidate_gen_vtime = cache->n_req;
    return cache->to_evict_candidate;
}

static void TLCacheMAB_evict(cache_t *cache, const request_t *req) {
    auto *params = static_cast<TLCacheMAB_params_t *>(cache->eviction_params);
    auto *mab    = static_cast<TLCache::TLCacheMABCache *>(params->TLCacheMAB_cache);

    if (cache->to_evict_candidate_gen_vtime == cache->n_req) {
        mab->evict_with_candidate(params->to_evict_pair);
        cache->to_evict_candidate_gen_vtime = -1;
    } else {
        mab->evict();
    }
}

static bool TLCacheMAB_remove(cache_t *cache, const obj_id_t obj_id) {
    ERROR("TLCacheMAB does not support remove\n");
    return true;
}

static int64_t TLCacheMAB_get_n_obj(const cache_t *cache) {
    auto *params = static_cast<TLCacheMAB_params_t *>(cache->eviction_params);
    auto *mab    = static_cast<TLCache::TLCacheMABCache *>(params->TLCacheMAB_cache);
    return (int64_t)mab->in_cache.metas.size();
}

static int64_t TLCacheMAB_get_occupied_byte(const cache_t *cache) {
    auto *params = static_cast<TLCacheMAB_params_t *>(cache->eviction_params);
    auto *mab    = static_cast<TLCache::TLCacheMABCache *>(params->TLCacheMAB_cache);
    return (int64_t)mab->_currentSize;
}

static void TLCacheMAB_parse_params(cache_t *cache, const char *cache_specific_params) {
    auto *params = static_cast<TLCacheMAB_params_t *>(cache->eviction_params);
    char *params_str = strdup(cache_specific_params);
    char *rest = params_str;

    while (rest != NULL && rest[0] != '\0') {
        char *key   = strsep(&rest, "=");
        char *value = strsep(&rest, ",");
        while (rest != NULL && *rest == ' ') rest++;

        if (strcasecmp(key, "objective") == 0) {
            free(params->objective);
            params->objective = strdup(value);
        } else if (strcasecmp(key, "arm_count") == 0 || strcasecmp(key, "arm-count") == 0) {
            params->arm_count = atoi(value);
        } else if (strcasecmp(key, "arm_strategy") == 0 || strcasecmp(key, "arm-strategy") == 0) {
            free(params->arm_strategy);
            params->arm_strategy = strdup(value);
        } else if (strcasecmp(key, "mab_gamma") == 0 || strcasecmp(key, "mab-gamma") == 0) {
            params->mab_gamma = atof(value);
        } else if (strcasecmp(key, "profiler_mode") == 0 || strcasecmp(key, "profiler-mode") == 0) {
            free(params->profiler_mode);
            params->profiler_mode = strdup(value);
        } else if (strcasecmp(key, "policy_file") == 0 || strcasecmp(key, "policy-file") == 0) {
            free(params->policy_file);
            params->policy_file = strdup(value);
        } else if (strcasecmp(key, "config_file") == 0 || strcasecmp(key, "config-file") == 0) {
            free(params->config_file);
            params->config_file = strdup(value);
        } else if (strcasecmp(key, "diag_enable") == 0 || strcasecmp(key, "diag-enable") == 0) {
            if (strcasecmp(value, "1") == 0 || strcasecmp(value, "true") == 0 || strcasecmp(value, "yes") == 0)
                params->diag_enable = 1;
            else if (strcasecmp(value, "0") == 0 || strcasecmp(value, "false") == 0 || strcasecmp(value, "no") == 0)
                params->diag_enable = 0;
            else
                params->diag_enable = -1;
        } else if (strcasecmp(key, "diag_interval") == 0 || strcasecmp(key, "diag-interval") == 0) {
            params->diag_interval = (uint64_t)strtoull(value, nullptr, 10);
            if (params->diag_interval == 0) params->diag_interval = 10000;
        } else if (strcasecmp(key, "diag_prefix") == 0 || strcasecmp(key, "diag-prefix") == 0) {
            free(params->diag_prefix);
            params->diag_prefix = strdup(value);
        } else if (strcasecmp(key, "train_log_enable") == 0 || strcasecmp(key, "train-log-enable") == 0) {
            if (strcasecmp(value, "1") == 0 || strcasecmp(value, "true") == 0 || strcasecmp(value, "yes") == 0)
                params->train_log_enable = 1;
            else if (strcasecmp(value, "0") == 0 || strcasecmp(value, "false") == 0 || strcasecmp(value, "no") == 0)
                params->train_log_enable = 0;
            else
                params->train_log_enable = -1;
        } else if (strcasecmp(key, "train_log_interval") == 0 || strcasecmp(key, "train-log-interval") == 0) {
            params->train_log_interval = (uint64_t)strtoull(value, nullptr, 10);
            if (params->train_log_interval == 0) params->train_log_interval = 10000;
        } else if (strcasecmp(key, "train_log_prefix") == 0 || strcasecmp(key, "train-log-prefix") == 0) {
            free(params->train_log_prefix);
            params->train_log_prefix = strdup(value);
        } else if (strcasecmp(key, "mab_off") == 0 || strcasecmp(key, "mab-off") == 0) {
            params->mab_off = (strcasecmp(value, "1") == 0 || strcasecmp(value, "true") == 0 ||
                               strcasecmp(value, "yes") == 0) ? 1 : 0;
        }
    }
    free(params_str);
}

#ifdef __cplusplus
}
#endif