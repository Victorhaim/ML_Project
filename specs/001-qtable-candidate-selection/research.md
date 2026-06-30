# Research: Q-Table Candidate Selection for 3L-Cache

**Phase 0 output for `/speckit-plan`**

---

## Technical Context (resolved)

| Field | Decision | Notes |
|-------|----------|-------|
| Language | C++17 | Same as TLCache.cpp / TLCache.h |
| Python scripts | Python 3 | Evaluation scripts under `3LCache/scripts/` |
| Build | Docker (Ubuntu 18.04, cmake 3.28.6) | Required per constitution; project `dockerfile` at repo root |
| Primary dependency | LightGBM (`LightGBM/c_api.h`) | Inherited; GBM predictor is *kept* in the eviction path |
| Interface | libCacheSim C ABI (`cache_t` function pointers) | Pattern: `TLCache_Interface.cpp`; new file follows same pattern |
| Test mechanism | `miss_ratio_boxplot.py` + `cpu_overhead_boxplot.py` | Both scripts must include new variant name |
| Naming convention | `TLCacheQL.{h,cpp}` + `TLCacheQL_Interface.cpp` | Follows existing `TLCacheX.{h,cpp}` + `TLCacheX_Interface.cpp` pattern |
| Algorithm string | `"3lcacheql"` | Follows `"3lcache"` convention; registered in `cache_struct_init` call |

---

## Decision 1: New variant inherits via C++ subclassing

**Decision**: `TLCacheQLCache` inherits from `TLCache::TLCacheCache` and overrides only `rank()`.

**Rationale**: All other methods (`lookup`, `admit`, `evict`, `evict_with_candidate`, `evict_predobj`, `prediction`, `train`, `sample`, `quick_demotion`) are unchanged. `rank()` is the sole entry point for the heuristic candidate-selection logic — it is the exact function to replace with the Q-table policy. Overriding only `rank()` satisfies constitution principles V (simplicity) and I (reproducibility).

**Alternatives considered**:
- Override `evict()` directly: rejected — too coarse; would require duplicating the booster-check + evict_predobj path.
- Composition over inheritance: rejected — the spec says "inherits from 3LCache"; subclassing is idiomatic here and the simplest approach.

---

## Decision 2: Q-table data structure

**Decision**: `std::unordered_map<uint32_t, std::array<float, 2>> q_table`

- Key: encoded state (see Decision 3).
- Value: `[Q(s, skip), Q(s, nominate)]` — two Q-values per state, one per action.
- Default value for unseen states: `{0.0f, 0.0f}` (treat as equal; fall back to LRU selection).

**Rationale**: `unordered_map` gives O(1) amortized lookup. Two-element array avoids allocating a separate structure. The state space is bounded (see Decision 3), so the map stays small.

**Alternatives considered**:
- Flat array indexed by state: requires knowing the maximum state key at compile time; bucketing parameters may be tuned, so a map is safer.
- Multi-armed bandit per state: equivalent when γ=0; simpler but Decision 4 justifies keeping γ as a tunable parameter.

---

## Decision 3: State encoding

**Decision**: State = `(freq_bucket << 6) | (age_bucket << 3) | size_bucket`; each bucket index is 3 bits → 8 buckets per dimension → 512 possible states.

| Dimension | Source field | Bucketing | Buckets |
|-----------|-------------|-----------|---------|
| Frequency | `meta._freq` | `min(7, (uint8_t)log2(freq))` | 8 |
| Age (recency) | `current_seq - meta._past_timestamp` | Quintile of `initial_queue_length`; 8 equal-width log2 windows | 8 |
| Size | `meta._size` | `min(7, (uint8_t)log2(size / 1024 + 1))` | 8 |

Encoded state fits in a `uint32_t` (only 9 bits used). The 3-bit frequency bucketing aligns with `object_distribution_n_eviction[uint16_t(log2(freq))]` already computed in `rank()`, so no extra computation.

**Rationale**: 512 states is small enough that the `unordered_map` never holds more than 512 entries (3-4 KB for floats). Lookup is O(1) and < 10 ns.

**Alternatives considered**:
- 4-bit buckets (16 per dimension, 4096 states): marginally more expressive; rejected for v1 to stay clearly within overhead budget.
- Including queue-position: rejected — per clarification Q3.

---

## Decision 4: Q-learning update rule

**Decision**: Episodic Q-learning, no discount (γ = 0), learning rate α = 0.1.

```
Q(s, a) ← Q(s, a) + α * (r - Q(s, a))
         = (1 - α) * Q(s, a) + α * r
```

Reward `r`:
- **−1** when the evicted object's key is seen as a hit in `out_cache` within the history window (i.e., `key_map[key].list_idx == 1` on re-request) — detected inside the inherited `lookup()` call that increments `n_window_hit`.
- **+1** when the object is NOT re-requested by the time it ages out of `out_cache`.

Positive terminal reward is recorded lazily: when `out_cache` overflows and the oldest entry is dropped without having triggered a penalty, it receives `+1`.

**Rationale**: γ = 0 simplifies to a running-average estimator per state-action pair. This is appropriate for an online eviction policy where the "episode" is a single eviction decision and future eviction decisions are weakly coupled. α = 0.1 is a standard conservative choice that is stable without warm-up.

**Alternatives considered**:
- Full Q-learning with γ > 0: requires defining "next state" after an eviction, which is ambiguous in this context; deferred to future work.
- UCB1 exploration: harder to implement without per-state visit counts; ε-greedy is sufficient for v1.

---

## Decision 5: Exploration policy

**Decision**: ε-greedy with ε = 0.1 (10% random nomination, 90% greedy Q-table choice).

During cold start (before the GBM booster is trained, i.e., `!booster`), the inherited `rank()` fallback is used unchanged (LRU-head). Once `booster != nullptr`, the Q-table sampling activates. This mirrors 3LCache's own LRU fallback pattern.

**Rationale**: Simple, proven for tabular RL. ε = 0.1 keeps enough exploration for states the cache hasn't visited recently without degrading performance noticeably.

---

## Decision 6: Reward detection mechanism

**Decision**: Add a thin `lookup()` override in `TLCacheQLCache` that intercepts re-requests of objects currently in `out_cache` to record pending-reward updates. The evicted object's key, its state-at-eviction-time, and the action taken are stored in a bounded `std::unordered_map<uint64_t, PendingUpdate>` (key → {state, action}).

When `lookup()` detects `key_map[key].list_idx == 1` (object is in `out_cache`), it:
1. Calls the base `lookup()`.
2. If a `PendingUpdate` entry exists for that key, applies the −1 reward update and removes the entry.

When `out_cache` evicts an entry (ages out, detected by `erase_out_cache`), if a `PendingUpdate` entry still exists, applies the +1 reward and removes the entry.

**Rationale**: The existing `out_cache` is the exact out-of-cache history window referenced in the spec. Intercepting `lookup()` is minimally invasive (one extra map lookup per request).

---

## Decision 7: libCacheSim registration and directory layout

**Decision**: The three new files (`TLCacheQL.h`, `TLCacheQL.cpp`, `TLCacheQL_Interface.cpp`) live in a new `3LCacheQL/` directory at the repo root, not inside `3LCache/`. `TLCacheQL_Interface.cpp` registers the algorithm as `"TLCacheQL"` via `cache_struct_init("TLCacheQL", ...)`. The script-level name is `"3lcacheql"` (lowercase, matching the existing `"3lcache"` string in Python scripts and executor).

CMakeLists.txt requires one additive glob block:
```cmake
file(GLOB TLCacheQL_source ./3LCacheQL/*.cpp)
set(cache_source ${cache_source} ${TLCacheQL_source})
```
The `TLCache.h` base class is included as `"../3LCache/TLCache.h"` from within `3LCacheQL/`.

**Rationale**: Each algorithm variant in this project lives in its own directory (consistent with `3LCache/`). Keeping `3LCacheQL/` separate avoids polluting `3LCache/` with a different algorithm's files and makes the boundary between base and variant explicit. The CMake change is strictly additive.

---

## Decision 8: Q-table statistics for SC-006 validation

**Decision**: Add a `q_table_updates` counter to `TLCacheQLCache` (incremented on every Q-update). Expose it via `update_stat_periodic()` override that logs it alongside the inherited OMR/BMR stats.

**Rationale**: SC-006 requires observable evidence that the Q-table learns. A cumulative update count that grows over the run is the simplest verifiable signal.
