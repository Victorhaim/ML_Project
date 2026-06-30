# Data Model: Q-Table Candidate Selection

## Entities

### 1. `QState` (value type, encoded as `uint32_t`)

Represents the discretized eviction-relevant description of one cached object.

| Field | Source | Encoding | Bits |
|-------|--------|----------|------|
| `freq_bucket` | `Meta::_freq` | `min(7, (uint8_t)floor(log2(freq)))` | 3 |
| `age_bucket` | `current_seq - Meta::_past_timestamp` | 8 equal log2-width windows over `[0, initial_queue_length]` | 3 |
| `size_bucket` | `Meta::_size` | `min(7, (uint8_t)floor(log2(size_kb + 1)))` where `size_kb = size/1024` | 3 |

Packed: `state = (freq_bucket << 6) | (age_bucket << 3) | size_bucket`

Total distinct states: 8 × 8 × 8 = **512**. Fits in `uint32_t` (9 bits used).

**Constraints**:
- All three dimensions must resolve to a value in [0, 7].
- The same encoding is applied consistently at candidate-selection time and at reward-update time.

---

### 2. `QEntry` (value in Q-table)

Stores the learned Q-values for one state.

| Field | Type | Semantics |
|-------|------|-----------|
| `q_skip` | `float` | Expected reward for action = "do not nominate this object as a candidate" |
| `q_nominate` | `float` | Expected reward for action = "include this object in the candidate set" |

Initial value: `{0.0f, 0.0f}`. Missing key = treat as `{0.0f, 0.0f}`.

**Update rule** (Q-learning, γ = 0):
```
Q(s, a) ← (1 − α) × Q(s, a) + α × r
```
Where α = 0.1 and r ∈ {−1.0f, +1.0f}.

---

### 3. `PendingUpdate`

Tracks the outcome of an eviction decision until a reward signal is received.

| Field | Type | Semantics |
|-------|------|-----------|
| `state` | `uint32_t` (QState) | State of the object at the time it was nominated and evicted |
| `action` | `uint8_t` | Always 1 (nominate); only nominated objects are evicted, so only they need reward tracking |

**Lifecycle**:
1. Created when an object is evicted after being Q-table-nominated.
2. Resolved with `r = −1` when the object's key is re-requested while still in `out_cache`.
3. Resolved with `r = +1` when the object ages out of `out_cache` without being re-requested.

**Storage**: `std::unordered_map<uint64_t, PendingUpdate>` keyed by object `key`. Bounded: at most `out_cache.metas.size()` entries, which is bounded by the configured history window size (same bound as the inherited 3LCache `out_cache`).

---

### 4. `TLCacheQLCache` (the new cache class)

Extends `TLCache::TLCacheCache`. Adds the following fields:

| Field | Type | Semantics |
|-------|------|-----------|
| `q_table` | `unordered_map<uint32_t, array<float,2>>` | The Q-table: state → [q_skip, q_nominate] |
| `pending_updates` | `unordered_map<uint64_t, PendingUpdate>` | Evicted-object → pending reward |
| `q_alpha` | `float` (default 0.1) | Learning rate |
| `q_epsilon` | `float` (default 0.1) | Exploration rate (ε-greedy) |
| `q_table_updates` | `uint64_t` | Cumulative Q-update count (for SC-006 observability) |

**Inherited fields used by the Q-table logic** (read-only access, not modified):
- `in_cache` — cache queue for candidate scanning
- `out_cache` / `key_map` — history window for reward detection
- `current_seq` — timestamp for age computation
- `initial_queue_length` — denominator for age bucketing
- `booster` — NULL-check drives cold-start fallback

---

## State Transitions

### Eviction candidate lifecycle

```
Object in in_cache
    │
    ▼ rank() called
Q-table evaluates (state, nominate) vs (state, skip)
    │
    ├─ skip → object stays in in_cache, no PendingUpdate created
    │
    └─ nominate → object added to candidate set
                  │
                  ▼ prediction() ranks candidates (GBM)
            Worst-ranked candidate evicted
                  │
                  ▼ evict_with_candidate()
            Object moves in_cache → out_cache
            PendingUpdate created: {state, action=nominate}
                  │
                  ├─ re-requested while in out_cache
                  │     → reward = −1, PendingUpdate removed
                  │
                  └─ ages out of out_cache without re-request
                        → reward = +1, PendingUpdate removed
```

---

## Validation Rules

- `QState` encoding is deterministic: same object attributes always produce the same `uint32_t`.
- `pending_updates` never holds a key for an object currently in `in_cache`.
- `q_table_updates` is monotonically non-decreasing over the run.
- After each Q-update, `q_table[state][action]` changes by at most `α × (r - old_value)` ≤ `2α`.
