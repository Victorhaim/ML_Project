
#include <assert.h>
#include <map>
#include <string>

#include "../libCacheSim/dataStructure/hashtable/hashtable.h"
#include "../libCacheSim/include/libCacheSim/cache.h"
#include "TLCacheQL.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  void   *TLCacheQL_cache;
  char   *objective;
  float   q_alpha;
  float   q_epsilon;
  SimpleRequest             TLCacheQL_req;
  pair<uint64_t, uint32_t> to_evict_pair;
  cache_obj_t               obj_tmp;
} TLCacheQL_params_t;

static const char *DEFAULT_PARAMS = "objective=byte-miss-ratio";

// ***********************************************************************
// ****                   function declarations                       ****
// ***********************************************************************

static void       TLCacheQL_free(cache_t *cache);
static bool       TLCacheQL_get(cache_t *cache, const request_t *req);
static cache_obj_t *TLCacheQL_find(cache_t *cache, const request_t *req,
                                   const bool update_cache);
static cache_obj_t *TLCacheQL_insert(cache_t *cache, const request_t *req);
static cache_obj_t *TLCacheQL_to_evict(cache_t *cache, const request_t *req);
static void       TLCacheQL_evict(cache_t *cache, const request_t *req);
static bool       TLCacheQL_remove(cache_t *cache, const obj_id_t obj_id);
static int64_t    TLCacheQL_get_occupied_byte(const cache_t *cache);
static int64_t    TLCacheQL_get_n_obj(const cache_t *cache);
static void       TLCacheQL_parse_params(cache_t *cache,
                                         const char *cache_specific_params);

// ***********************************************************************
// ****                   init / free / get                           ****
// ***********************************************************************

cache_t *TLCacheQL_init(const common_cache_params_t ccache_params,
                        const char *cache_specific_params) {
#ifdef SUPPORT_TTL
  if (ccache_params.default_ttl < 30 * 86400) {
    ERROR("TLCacheQL does not support expiration\n");
    abort();
  }
#endif

  cache_t *cache = cache_struct_init("TLCacheQL", ccache_params,
                                     cache_specific_params);
  cache->cache_init        = TLCacheQL_init;
  cache->cache_free        = TLCacheQL_free;
  cache->get               = TLCacheQL_get;
  cache->find              = TLCacheQL_find;
  cache->insert            = TLCacheQL_insert;
  cache->evict             = TLCacheQL_evict;
  cache->to_evict          = TLCacheQL_to_evict;
  cache->remove            = TLCacheQL_remove;
  cache->can_insert        = cache_can_insert_default;
  cache->get_occupied_byte = TLCacheQL_get_occupied_byte;
  cache->get_n_obj         = TLCacheQL_get_n_obj;
  cache->to_evict_candidate =
      static_cast<cache_obj_t *>(malloc(sizeof(cache_obj_t)));

  if (ccache_params.consider_obj_metadata) {
    cache->obj_md_size = 180;
  } else {
    cache->obj_md_size = 0;
  }

  auto *params = my_malloc(TLCacheQL_params_t);
  memset(params, 0, sizeof(TLCacheQL_params_t));
  params->q_alpha   = 0.1f;
  params->q_epsilon = 0.1f;
  cache->eviction_params = params;

  if (cache_specific_params != NULL) {
    TLCacheQL_parse_params(cache, cache_specific_params);
  } else {
    TLCacheQL_parse_params(cache, DEFAULT_PARAMS);
  }

  auto *qlcache = new TLCacheQL::TLCacheQLCache();
  params->TLCacheQL_cache = static_cast<void *>(qlcache);

  qlcache->setSize(ccache_params.cache_size);

  std::map<string, string> params_map;
  params_map["objective"] = params->objective;
  params_map["q_alpha"]   = std::to_string(params->q_alpha);
  params_map["q_epsilon"] = std::to_string(params->q_epsilon);

  if (strcmp(params->objective, "object-miss-ratio") == 0) {
    snprintf(cache->cache_name, CACHE_NAME_ARRAY_LEN, "%s", "TLCacheQL-OMR");
  } else if (strcasecmp(params->objective, "byte-miss-ratio") == 0) {
    snprintf(cache->cache_name, CACHE_NAME_ARRAY_LEN, "%s", "TLCacheQL-BMR");
  } else {
    ERROR("TLCacheQL does not support objective %s\n", params->objective);
  }

  qlcache->init_with_params(params_map);

  return cache;
}

static void TLCacheQL_free(cache_t *cache) {
  auto *params  = static_cast<TLCacheQL_params_t *>(cache->eviction_params);
  auto *qlcache = static_cast<TLCacheQL::TLCacheQLCache *>(params->TLCacheQL_cache);
  delete qlcache;
  free(cache->to_evict_candidate);
  my_free(sizeof(TLCacheQL_params_t), params);
  cache_struct_free(cache);
}

static bool TLCacheQL_get(cache_t *cache, const request_t *req) {
  return cache_get_base(cache, req);
}

// ***********************************************************************
// ****              developer-facing functions                        ****
// ***********************************************************************

static cache_obj_t *TLCacheQL_find(cache_t *cache, const request_t *req,
                                   const bool update_cache) {
  auto *params  = static_cast<TLCacheQL_params_t *>(cache->eviction_params);
  auto *qlcache = static_cast<TLCacheQL::TLCacheQLCache *>(params->TLCacheQL_cache);

  if (!update_cache) {
    bool is_hit = qlcache->exist(static_cast<int64_t>(req->obj_id));
    return is_hit ? reinterpret_cast<cache_obj_t *>(0x1) : NULL;
  }

  params->TLCacheQL_req.reinit(cache->n_req, req->obj_id, req->obj_size, nullptr);
  bool is_hit = qlcache->lookup(params->TLCacheQL_req);

  return is_hit ? reinterpret_cast<cache_obj_t *>(0x1) : NULL;
}

static cache_obj_t *TLCacheQL_insert(cache_t *cache, const request_t *req) {
  auto *params  = static_cast<TLCacheQL_params_t *>(cache->eviction_params);
  auto *qlcache = static_cast<TLCacheQL::TLCacheQLCache *>(params->TLCacheQL_cache);

  params->TLCacheQL_req.reinit(cache->n_req, req->obj_id, req->obj_size, nullptr);
  qlcache->admit(params->TLCacheQL_req);

  return reinterpret_cast<cache_obj_t *>(0x1);
}

static cache_obj_t *TLCacheQL_to_evict(cache_t *cache, const request_t *req) {
  auto *params  = static_cast<TLCacheQL_params_t *>(cache->eviction_params);
  auto *qlcache = static_cast<TLCacheQL::TLCacheQLCache *>(params->TLCacheQL_cache);

  if (!qlcache->booster) {
    // Cold start: LRU head
    uint32_t pos = qlcache->in_cache.q.head;
    params->to_evict_pair = {qlcache->in_cache.metas[pos]._key, pos};
  } else {
    if (qlcache->evict_nums <= 0 || qlcache->pred_map.empty()) {
      qlcache->evict_nums = (int)(qlcache->ql_rank() / qlcache->eviction_rate);
    }
    params->to_evict_pair = qlcache->pop_victim();
    if (params->to_evict_pair.first == (uint64_t)-1) {
      uint32_t pos = qlcache->in_cache.q.head;
      params->to_evict_pair = {qlcache->in_cache.metas[pos]._key, pos};
    }
  }

  auto &meta = qlcache->in_cache.metas[params->to_evict_pair.second];
  params->obj_tmp.obj_id   = params->to_evict_pair.first;
  params->obj_tmp.obj_size = meta._size;

  cache->to_evict_candidate = &params->obj_tmp;
  cache->to_evict_candidate_gen_vtime = cache->n_req;

  return cache->to_evict_candidate;
}

static void TLCacheQL_evict(cache_t *cache, const request_t *req) {
  auto *params  = static_cast<TLCacheQL_params_t *>(cache->eviction_params);
  auto *qlcache = static_cast<TLCacheQL::TLCacheQLCache *>(params->TLCacheQL_cache);

  if (cache->to_evict_candidate_gen_vtime == cache->n_req) {
    qlcache->evict_with_candidate_tracking(params->to_evict_pair);
    cache->to_evict_candidate_gen_vtime = -1;
  } else {
    qlcache->evict();
  }
}

static bool TLCacheQL_remove(cache_t *cache, const obj_id_t obj_id) {
  ERROR("do not support remove");
  return true;
}

static int64_t TLCacheQL_get_n_obj(const cache_t *cache) {
  auto *params  = static_cast<TLCacheQL_params_t *>(cache->eviction_params);
  auto *qlcache = static_cast<TLCacheQL::TLCacheQLCache *>(params->TLCacheQL_cache);
  return (int64_t)qlcache->in_cache.metas.size();
}

static int64_t TLCacheQL_get_occupied_byte(const cache_t *cache) {
  auto *params  = static_cast<TLCacheQL_params_t *>(cache->eviction_params);
  auto *qlcache = static_cast<TLCacheQL::TLCacheQLCache *>(params->TLCacheQL_cache);
  return qlcache->_currentSize;
}

// ***********************************************************************
// ****                  parameter parsing                            ****
// ***********************************************************************

static void TLCacheQL_parse_params(cache_t *cache,
                                   const char *cache_specific_params) {
  TLCacheQL_params_t *params =
      (TLCacheQL_params_t *)cache->eviction_params;
  char *params_str = strdup(cache_specific_params);

  while (params_str != NULL && params_str[0] != '\0') {
    char *key   = strsep((char **)&params_str, "=");
    char *value = strsep((char **)&params_str, ",");

    while (params_str != NULL && *params_str == ' ')
      params_str++;

    if (strcasecmp(key, "objective") == 0) {
      params->objective = strdup(value);
      if (params->objective == NULL)
        ERROR("out of memory %s\n", strerror(errno));
    } else if (strcasecmp(key, "q_alpha") == 0) {
      params->q_alpha = (float)atof(value);
    } else if (strcasecmp(key, "q_epsilon") == 0) {
      params->q_epsilon = (float)atof(value);
    } else if (strcasecmp(key, "print") == 0) {
      printf("current parameters: objective=%s,q_alpha=%f,q_epsilon=%f\n",
             params->objective, params->q_alpha, params->q_epsilon);
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
