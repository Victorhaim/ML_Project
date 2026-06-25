// TLCacheMAB_Interface.cpp
//
// C interface between libCacheSim and the TLCacheMABCache C++ class.
// Follows the same pattern as TLCache_Interface.cpp.

#include <assert.h>
#include <map>
#include <string>

#include "../libCacheSim/dataStructure/hashtable/hashtable.h"
#include "../libCacheSim/include/libCacheSim/cache.h"
#include "TLCacheMAB.h"

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------
// Parameter struct stored in cache->eviction_params
// -----------------------------------------------------------------------
typedef struct {
    void            *TLCacheMAB_cache;
    char            *objective;
    int              arm_count;
    char            *arm_strategy;
    double           mab_gamma;
    SimpleRequest    TLCacheMAB_req;
    pair<uint64_t, uint32_t> to_evict_pair;
    cache_obj_t      obj_tmp;
} TLCacheMAB_params_t;

static const char *DEFAULT_PARAMS =
    "objective=byte-miss-ratio,arm_count=4,arm_strategy=position,mab_gamma=0.1";

// -----------------------------------------------------------------------
// Forward declarations
// -----------------------------------------------------------------------
static void    TLCacheMAB_free(cache_t *cache);
static bool    TLCacheMAB_get(cache_t *cache, const request_t *req);
static cache_obj_t *TLCacheMAB_find(cache_t *cache, const request_t *req,
                                    const bool update_cache);
static cache_obj_t *TLCacheMAB_insert(cache_t *cache, const request_t *req);
static cache_obj_t *TLCacheMAB_to_evict(cache_t *cache, const request_t *req);
static void    TLCacheMAB_evict(cache_t *cache, const request_t *req);
static bool    TLCacheMAB_remove(cache_t *cache, const obj_id_t obj_id);
static int64_t TLCacheMAB_get_occupied_byte(const cache_t *cache);
static int64_t TLCacheMAB_get_n_obj(const cache_t *cache);
static void    TLCacheMAB_parse_params(cache_t *cache,
                                       const char *cache_specific_params);

// -----------------------------------------------------------------------
// TLCacheMAB_init
// -----------------------------------------------------------------------
cache_t *TLCacheMAB_init(const common_cache_params_t ccache_params,
                          const char *cache_specific_params) {
#ifdef SUPPORT_TTL
    if (ccache_params.default_ttl < 30 * 86400) {
        ERROR("TLCacheMAB does not support expiration\n");
        abort();
    }
#endif

    cache_t *cache = cache_struct_init("TLCacheMAB", ccache_params,
                                       cache_specific_params);
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
    cache->to_evict_candidate  =
        static_cast<cache_obj_t *>(malloc(sizeof(cache_obj_t)));

    if (ccache_params.consider_obj_metadata) {
        cache->obj_md_size = 180;
    } else {
        cache->obj_md_size = 0;
    }

    auto *params = my_malloc(TLCacheMAB_params_t);
    memset(params, 0, sizeof(TLCacheMAB_params_t));
    cache->eviction_params = params;

    // Set defaults before parsing
    params->objective    = strdup("byte-miss-ratio");
    params->arm_count    = 4;
    params->arm_strategy = strdup("position");
    params->mab_gamma    = 0.1;

    if (cache_specific_params != NULL) {
        TLCacheMAB_parse_params(cache, cache_specific_params);
    } else {
        TLCacheMAB_parse_params(cache, DEFAULT_PARAMS);
    }

    // Construct the C++ MAB cache object
    auto *mab = new TLCache::TLCacheMABCache();
    params->TLCacheMAB_cache = static_cast<void *>(mab);
    mab->setSize(ccache_params.cache_size);

    // Build the params map passed to init_with_params
    std::map<std::string, std::string> params_map;
    params_map["objective"]    = params->objective;
    params_map["arm_count"]    = std::to_string(params->arm_count);
    params_map["arm_strategy"] = params->arm_strategy;
    params_map["mab_gamma"]    = std::to_string(params->mab_gamma);

    mab->init_with_params(params_map);

    // Set cache name shown in results
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

// -----------------------------------------------------------------------
// TLCacheMAB_free
// -----------------------------------------------------------------------
static void TLCacheMAB_free(cache_t *cache) {
    auto *params = static_cast<TLCacheMAB_params_t *>(cache->eviction_params);
    auto *mab    = static_cast<TLCache::TLCacheMABCache *>(
                       params->TLCacheMAB_cache);
    delete mab;
    free(params->objective);
    free(params->arm_strategy);
    free(cache->to_evict_candidate);
    my_free(sizeof(TLCacheMAB_params_t), params);
    cache_struct_free(cache);
}

// -----------------------------------------------------------------------
// TLCacheMAB_get
// -----------------------------------------------------------------------
static bool TLCacheMAB_get(cache_t *cache, const request_t *req) {
    return cache_get_base(cache, req);
}

// -----------------------------------------------------------------------
// TLCacheMAB_find
// -----------------------------------------------------------------------
static cache_obj_t *TLCacheMAB_find(cache_t *cache, const request_t *req,
                                    const bool update_cache) {
    auto *params = static_cast<TLCacheMAB_params_t *>(cache->eviction_params);
    auto *mab    = static_cast<TLCache::TLCacheMABCache *>(
                       params->TLCacheMAB_cache);

    if (!update_cache) {
        bool is_hit = mab->exist(static_cast<int64_t>(req->obj_id));
        return is_hit ? reinterpret_cast<cache_obj_t *>(0x1) : NULL;
    }

    params->TLCacheMAB_req.reinit(cache->n_req, req->obj_id,
                                   req->obj_size, nullptr);
    bool is_hit = mab->lookup(params->TLCacheMAB_req);
    return is_hit ? reinterpret_cast<cache_obj_t *>(0x1) : NULL;
}

// -----------------------------------------------------------------------
// TLCacheMAB_insert
// -----------------------------------------------------------------------
static cache_obj_t *TLCacheMAB_insert(cache_t *cache, const request_t *req) {
    auto *params = static_cast<TLCacheMAB_params_t *>(cache->eviction_params);
    auto *mab    = static_cast<TLCache::TLCacheMABCache *>(
                       params->TLCacheMAB_cache);
    params->TLCacheMAB_req.reinit(cache->n_req, req->obj_id,
                                   req->obj_size, nullptr);
    mab->admit(params->TLCacheMAB_req);
    return reinterpret_cast<cache_obj_t *>(0x1);
}

// -----------------------------------------------------------------------
// TLCacheMAB_to_evict
// -----------------------------------------------------------------------
static cache_obj_t *TLCacheMAB_to_evict(cache_t *cache, const request_t *req) {
    auto *params = static_cast<TLCacheMAB_params_t *>(cache->eviction_params);
    auto *mab    = static_cast<TLCache::TLCacheMABCache *>(
                       params->TLCacheMAB_cache);

    params->to_evict_pair = mab->evict_predobj();
    auto &meta = mab->in_cache.metas[params->to_evict_pair.second];

    params->obj_tmp.obj_id   = params->to_evict_pair.first;
    params->obj_tmp.obj_size = meta._size;

    cache->to_evict_candidate = &params->obj_tmp;
    cache->to_evict_candidate_gen_vtime = cache->n_req;
    return cache->to_evict_candidate;
}

// -----------------------------------------------------------------------
// TLCacheMAB_evict
// -----------------------------------------------------------------------
static void TLCacheMAB_evict(cache_t *cache, const request_t *req) {
    auto *params = static_cast<TLCacheMAB_params_t *>(cache->eviction_params);
    auto *mab    = static_cast<TLCache::TLCacheMABCache *>(
                       params->TLCacheMAB_cache);

    if (cache->to_evict_candidate_gen_vtime == cache->n_req) {
        mab->evict_with_candidate(params->to_evict_pair);
        cache->to_evict_candidate_gen_vtime = -1;
    } else {
        mab->evict();
    }
}

// -----------------------------------------------------------------------
// TLCacheMAB_remove
// -----------------------------------------------------------------------
static bool TLCacheMAB_remove(cache_t *cache, const obj_id_t obj_id) {
    ERROR("TLCacheMAB does not support remove\n");
    return true;
}

// -----------------------------------------------------------------------
// Utility: size / count
// -----------------------------------------------------------------------
static int64_t TLCacheMAB_get_n_obj(const cache_t *cache) {
    auto *params = static_cast<TLCacheMAB_params_t *>(cache->eviction_params);
    auto *mab    = static_cast<TLCache::TLCacheMABCache *>(
                       params->TLCacheMAB_cache);
    return (int64_t)mab->in_cache.metas.size();
}

static int64_t TLCacheMAB_get_occupied_byte(const cache_t *cache) {
    auto *params = static_cast<TLCacheMAB_params_t *>(cache->eviction_params);
    auto *mab    = static_cast<TLCache::TLCacheMABCache *>(
                       params->TLCacheMAB_cache);
    return (int64_t)mab->_currentSize;
}

// -----------------------------------------------------------------------
// Parameter parsing
// -----------------------------------------------------------------------
static void TLCacheMAB_parse_params(cache_t *cache,
                                    const char *cache_specific_params) {
    auto *params = static_cast<TLCacheMAB_params_t *>(cache->eviction_params);
    char *params_str = strdup(cache_specific_params);
    char *rest = params_str;

    while (rest != NULL && rest[0] != '\0') {
        char *key   = strsep(&rest, "=");
        char *value = strsep(&rest, ",");
        // Skip leading spaces
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
        } else if (strcasecmp(key, "print") == 0) {
            printf("TLCacheMAB params: objective=%s arm_count=%d "
                   "arm_strategy=%s mab_gamma=%.3f\n",
                   params->objective, params->arm_count,
                   params->arm_strategy, params->mab_gamma);
            exit(0);
        } else {
            ERROR("%s does not have parameter %s\n", cache->cache_name, key);
            exit(1);
        }
    }
    free(params_str);
}

#ifdef __cplusplus
}
#endif
