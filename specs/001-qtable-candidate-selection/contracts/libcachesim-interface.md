# Contract: libCacheSim C Interface for TLCacheQL

The new variant must satisfy the same libCacheSim `cache_t` function-pointer contract as the existing `TLCache`. This contract is defined by `libCacheSim/include/libCacheSim/cache.h` and implemented in `TLCacheQL_Interface.cpp`.

## Required function-pointer bindings

| Slot | Function signature | Behaviour contract |
|------|-------------------|--------------------|
| `cache_init` | `cache_t *TLCacheQL_init(common_cache_params_t, const char*)` | Allocates and returns an initialized `cache_t`; sets all function pointers; calls `TLCacheQLCache->init_with_params()` |
| `cache_free` | `void TLCacheQL_free(cache_t*)` | Deletes the `TLCacheQLCache` object and frees `eviction_params` |
| `get` | `bool TLCacheQL_get(cache_t*, const request_t*)` | Delegates to `cache_get_base(cache, req)` (unchanged from `TLCache`) |
| `find` | `cache_obj_t *TLCacheQL_find(cache_t*, const request_t*, bool)` | Looks up object; if found and `update_cache=true`, promotes it; triggers Q-table reward update if object is in `out_cache` |
| `insert` | `cache_obj_t *TLCacheQL_insert(cache_t*, const request_t*)` | Admits a new object; unchanged from `TLCache` |
| `evict` | `void TLCacheQL_evict(cache_t*, const request_t*)` | Calls `TLCacheQLCache->evict()` → same call chain as `TLCache`, but `rank()` is overridden |
| `to_evict` | `cache_obj_t *TLCacheQL_to_evict(cache_t*, const request_t*)` | Returns eviction victim without removing; delegates to `evict_predobj()` (inherited) |
| `remove` | `bool TLCacheQL_remove(cache_t*, obj_id_t)` | Removes a specific object; unchanged from `TLCache` |
| `get_occupied_byte` | `int64_t TLCacheQL_get_occupied_byte(const cache_t*)` | Returns `_currentSize`; unchanged |
| `get_n_obj` | `int64_t TLCacheQL_get_n_obj(const cache_t*)` | Returns `in_cache.metas.size()`; unchanged |

## Cache name

`cache_struct_init("TLCacheQL", ccache_params, cache_specific_params)`

This sets `cache->cache_name` to `"TLCacheQL-BMR"` or `"TLCacheQL-OMR"` depending on the `objective` parameter, consistent with the `TLCache` pattern.

## Parameters

The variant accepts the same parameters as `TLCache` (forwarded to `init_with_params`). Additional optional parameters:

| Parameter | Type | Default | Semantics |
|-----------|------|---------|-----------|
| `q_alpha` | float | 0.1 | Q-learning rate |
| `q_epsilon` | float | 0.1 | ε-greedy exploration rate |

Parameters are parsed in `TLCacheQL_parse_params` following the same pattern as `TLCache_parse_params`.

## File location

`TLCacheQL.h`, `TLCacheQL.cpp`, and `TLCacheQL_Interface.cpp` live in `3LCacheQL/` at the repo root (not inside `3LCache/`). The base class header is included as `"../3LCache/TLCache.h"`. CMakeLists.txt adds one glob block for `./3LCacheQL/*.cpp`.

## Registration in evaluation scripts

The string `"3lcacheql"` must be added to the algorithm list in:

- `3LCache/scripts/executor_libcachesim.py` — lines 63 and 71 (the `cache_strategy` lists)
- `3LCache/scripts/miss_ratio_boxplot.py` — algorithm name mapping block (add `elif algo[:9] == 'TLCacheQL': key_map[algo] = '3L-Cache-QL'`)
- `3LCache/scripts/cpu_overhead_boxplot.py` — same mapping block if present

## Invariants

- The `cache_t*` pointer returned by `TLCacheQL_init` is always non-NULL.
- After every `evict` call, `get_occupied_byte() <= cache->cache_size`.
- The `find` function MUST NOT modify cache state when `update_cache = false`.
- Calling `get`, `find`, `insert`, `evict`, `remove` in any order must not corrupt `TLCacheQLCache` internal state.
