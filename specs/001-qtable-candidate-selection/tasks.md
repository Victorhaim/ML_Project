# Tasks: Q-Table Candidate Selection for 3L-Cache

**Input**: Design documents from `specs/001-qtable-candidate-selection/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/libcachesim-interface.md, quickstart.md

**Tests**: Not explicitly requested in spec — no test tasks generated. Validation follows quickstart.md.

**Organization**: Tasks are grouped by user story. US1 and US2 are both P1; US1 (variant runnable) must land before US2 (Q-table logic) because US2 builds on the same files. US3 (script comparison) is P2 and depends only on US1 completion.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no incomplete-task dependencies)
- **[Story]**: User story this task belongs to

---

## Phase 1: Setup

**Purpose**: Create the new variant's directory and wire it into the build system.

- [x] T001 Create `3LCacheQL/` directory at repository root
- [x] T002 Add `file(GLOB TLCacheQL_source ./3LCacheQL/*.cpp)` and `set(cache_source ${cache_source} ${TLCacheQL_source})` to `CMakeLists.txt` after the existing TLCache glob block (line ~329)

**Checkpoint**: `3LCacheQL/` directory exists; CMakeLists.txt will compile any `.cpp` files placed there.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Define the shared header that all three source files (`TLCacheQL.cpp`, `TLCacheQL_Interface.cpp`, and any future extensions) depend on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T003 Write `3LCacheQL/TLCacheQL.h` — declare `PendingUpdate` struct and `TLCacheQLCache` class inheriting from `TLCache::TLCacheCache`, with fields `q_table` (`unordered_map<uint32_t, array<float,2>>`), `pending_updates` (`unordered_map<uint64_t, PendingUpdate>`), `q_alpha` (float, default 0.1), `q_epsilon` (float, default 0.1), `q_table_updates` (uint64_t); declare overrides `init_with_params`, `lookup`, `rank`, `update_stat_periodic`; declare private helpers `encode_state` and `apply_reward`; include `"../3LCache/TLCache.h"`

**Checkpoint**: Foundation ready — `TLCacheQL.h` compiles cleanly when included.

---

## Phase 3: User Story 1 — Run the Variant in a Miss-Ratio Experiment (Priority: P1) 🎯 MVP

**Goal**: The new variant is registered with libCacheSim, compiles inside Docker, and produces valid OMR and BMR output on any standard trace — using the inherited LRU fallback for candidate selection until the Q-table is wired up in Phase 4.

**Independent Test**: Inside Docker, run `./_build/bin/cachesim <trace> csv -t "time-col=0,obj-id-col=1,obj-size-col=2" -a 3lcacheql -s <size> -e "objective=byte-miss-ratio"` — run completes, prints `TLCacheQL-BMR ... object miss ratio: X, byte miss ratio: Y`, no crash, cache size never violated (quickstart.md Step 2).

### Implementation for User Story 1

- [x] T004 [US1] Implement `3LCacheQL/TLCacheQL_Interface.cpp` — full libCacheSim C adapter following the pattern in `3LCache/TLCache_Interface.cpp`: define `TLCacheQL_params_t`, implement `TLCacheQL_init` (calls `cache_struct_init("TLCacheQL", ...)`, sets all `cache_t` function pointers, sets `cache_name` to `"TLCacheQL-BMR"` or `"TLCacheQL-OMR"` per objective), `TLCacheQL_free`, `TLCacheQL_get` (delegates to `cache_get_base`), `TLCacheQL_find`, `TLCacheQL_insert`, `TLCacheQL_evict`, `TLCacheQL_to_evict`, `TLCacheQL_remove`, `TLCacheQL_get_occupied_byte`, `TLCacheQL_get_n_obj`, `TLCacheQL_parse_params` (accepts same params as TLCache plus `q_alpha` and `q_epsilon`)

- [x] T005 [US1] Implement `3LCacheQL/TLCacheQL.cpp` — `init_with_params()` override: call `TLCacheCache::init_with_params(params)`, then parse `q_alpha` and `q_epsilon` from the params map; implement `rank()` override as a cold-start stub that always calls and returns `TLCacheCache::rank()` (full Q-table logic added in Phase 4); implement `update_stat_periodic()` override that calls the base and logs `q_table_updates` to stderr

- [ ] T006 [US1] Build the project inside Docker (`sudo docker build -t 3lcache -f dockerfile . && sudo docker run ... cmake .. && make -j$(nproc)`) and confirm `_build/bin/cachesim` is produced without errors; fix any compile errors in `3LCacheQL/` before proceeding

**Checkpoint**: US1 complete — `cachesim -a 3lcacheql` produces OMR and BMR on any trace. Baseline `3lcache` results are unchanged.

---

## Phase 4: User Story 2 — Candidate Selection Driven by the Q-Table (Priority: P1)

**Goal**: The `rank()` override replaces the inherited heuristic sampler with the Q-table policy. The Q-table updates online from eviction outcomes (re-request detection via `out_cache`). The GBM predictor is retained to rank the Q-table-selected candidates.

**Independent Test**: Run the variant on a trace with ≥ 1M requests; confirm stderr shows `q_table_updates` growing from 0 over the run; confirm the Q-table has non-zero entries after the run. Candidate selection must not stall (quickstart.md Step 3).

### Implementation for User Story 2

- [x] T007 [US2] Implement `encode_state(const Meta& meta) const` in `3LCacheQL/TLCacheQL.cpp` — compute `freq_bucket = min(7, (uint8_t)floor(log2(meta._freq)))`, `age_bucket = min(7, (uint8_t)(8 * (current_seq - meta._past_timestamp) / (initial_queue_length + 1)))`, `size_bucket = min(7, (uint8_t)floor(log2(meta._size / 1024 + 1)))`; return `(freq_bucket << 6) | (age_bucket << 3) | size_bucket` as `uint32_t`

- [x] T008 [US2] Replace the `rank()` stub in `3LCacheQL/TLCacheQL.cpp` with the full Q-table implementation: (1) cold-start guard — if `!booster`, call and return `TLCacheCache::rank()`; (2) call `quick_demotion()` (inherited); (3) walk cache queue from `samplepointer` for up to `sample_rate` steps — for each object call `encode_state`, look up `q_table` (default `{0,0}` if missing), apply ε-greedy (with prob `q_epsilon` choose random action, else choose action with higher Q-value), if action = nominate add pos to `sampled_objects`, advance `samplepointer` with wraparound; (4) fallback: if `sampled_objects` is empty, force-add `in_cache.q.head`; (5) call `prediction(sampled_objects)` (inherited); (6) return `sampled_objects.size()`

- [x] T009 [US2] Implement `apply_reward(uint64_t key, float reward)` in `3LCacheQL/TLCacheQL.cpp` — look up `pending_updates[key]`, retrieve `{state, action}`, apply `q_table[state][action] = (1 - q_alpha) * q_table[state][action] + q_alpha * reward`, increment `q_table_updates`, erase `pending_updates[key]`

- [x] T010 [US2] Implement `lookup()` override in `3LCacheQL/TLCacheQL.cpp` — call `TLCacheCache::lookup(req)`, then check if `key_map.count(req.id) && key_map[req.id].list_idx == 1` (object in `out_cache`, i.e., re-request of evicted object) and `pending_updates.count(req.id)` — if both true, call `apply_reward(req.id, -1.0f)`; return the base result; also, when an object is evicted (detected by intercepting after `evict_with_candidate` — add a hook to `admit()` or override the eviction path): create a `PendingUpdate{encode_state(meta_at_eviction_time), 1}` entry in `pending_updates` keyed by the evicted object's key

- [x] T011 [US2] Implement positive reward on `out_cache` age-out in `3LCacheQL/TLCacheQL.cpp` — override `erase_out_cache()` or hook into `admit()` where `out_cache` is trimmed: after `TLCacheCache::erase_out_cache()` is called (or the equivalent trim logic), for any key removed from `out_cache` that still has a `pending_updates` entry, call `apply_reward(key, +1.0f)`; note: examine `TLCache.cpp::erase_out_cache()` to understand the trim trigger and replicate the hook correctly

**Checkpoint**: US2 complete — Q-table updates accumulate during a long trace run, candidate selection is driven by the Q-table, inherited GBM predictor still ranks the nominated candidates, and the cold-start fallback works on a fresh cache.

---

## Phase 5: User Story 3 — Compare Against Baselines (Priority: P2)

**Goal**: The variant appears in the standard miss-ratio and CPU-overhead comparison scripts alongside baseline `3lcache` and the established baselines, producing box plots and result files that include `3lcacheql`.

**Independent Test**: Run `miss_ratio_boxplot.py` with `"3lcacheql"` in the `--algo` list; confirm a box plot and result CSV are produced containing a `3L-Cache-QL` column (quickstart.md Steps 4–6).

### Implementation for User Story 3

- [x] T012 [P] [US3] Add `"3lcacheql"` to both `cache_strategy` lists in `3LCache/scripts/executor_libcachesim.py` (lines 63 and 71)

- [x] T013 [P] [US3] Add `elif algo[:9] == 'TLCacheQL': key_map[algo] = '3L-Cache-QL'` to the algorithm name-mapping block in `3LCache/scripts/miss_ratio_boxplot.py` (after the existing `TLCache` mapping at line ~108)

- [x] T014 [P] [US3] Add `elif algo[:9] == 'TLCacheQL': key_map[algo] = '3L-Cache-QL'` to the equivalent algorithm name-mapping block in `3LCache/scripts/cpu_overhead_boxplot.py` (if a mapping block exists; otherwise add the entry inline in the plot label logic)

**Checkpoint**: US3 complete — running `miss_ratio_boxplot.py` and `cpu_overhead_boxplot.py` with `"3lcacheql"` included produces figures and result files that include `3L-Cache-QL`.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validate constitution compliance, verify no regressions, and confirm all success criteria.

- [ ] T015 [P] Run `cpu_overhead_boxplot.py` with `['3lcacheql', '3lcache', 'lru']` inside Docker and confirm `3lcacheql` overhead is within `3lcache`'s documented threshold (SC-005; constitution principle II gate)

- [ ] T016 [P] Run `miss_ratio_boxplot.py` with `['3lcache', 'lru']` only (no `3lcacheql`) and confirm `3lcache` BMR and OMR results are identical to pre-change baseline results (SC-004 no-regression check)

- [ ] T017 Complete the quickstart.md acceptance sign-off checklist end-to-end (all six checkboxes: SC-001 through SC-006)

- [ ] T018 Ensure commit message for the eviction-logic change includes "OMR" or "BMR" and a summary of the measured impact (constitution principle — commits touching eviction logic MUST include these words)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 — blocks all user story work
- **US1 (Phase 3)**: Depends on Phase 2
- **US2 (Phase 4)**: Depends on Phase 3 (adds to the same `TLCacheQL.cpp` file)
- **US3 (Phase 5)**: Depends on Phase 3 only (script changes; Q-table not required for scripts)
- **Polish (Phase 6)**: Depends on Phase 4 + Phase 5

### User Story Dependencies

- **US1 (P1)**: Blocked only by Foundational
- **US2 (P1)**: Blocked by US1 (shares `TLCacheQL.cpp`)
- **US3 (P2)**: Blocked by US1; can proceed in parallel with US2

### Within Each User Story

- T004 and T005 are parallelizable (different files: `TLCacheQL_Interface.cpp` vs `TLCacheQL.cpp`)
- T007 must complete before T008 (encode_state used in rank())
- T009 must complete before T010 and T011 (apply_reward used in both)
- T012, T013, T014 are all parallelizable (different script files)

### Parallel Opportunities

```bash
# Phase 3 — US1 (after T003 done):
Task T004: 3LCacheQL/TLCacheQL_Interface.cpp
Task T005: 3LCacheQL/TLCacheQL.cpp (stub)   ← run in parallel with T004

# Phase 5 — US3 (after T006 done):
Task T012: executor_libcachesim.py
Task T013: miss_ratio_boxplot.py             ← all three in parallel
Task T014: cpu_overhead_boxplot.py

# Phase 6 — Polish (after Phase 4 + Phase 5 done):
Task T015: cpu overhead validation
Task T016: no-regression check               ← run in parallel
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001–T002)
2. Complete Phase 2: Foundational (T003)
3. Complete Phase 3: US1 (T004–T006)
4. **STOP and VALIDATE**: Run `cachesim -a 3lcacheql` on one trace; confirm OMR and BMR output
5. Variant is registered and runnable — proceed to US2 for the Q-table logic

### Incremental Delivery

1. Setup + Foundational → header and build wired up
2. US1 → variant runs (LRU fallback), produces valid results
3. US2 → Q-table learning active, candidates selected by policy
4. US3 → variant appears in comparison figures
5. Polish → overhead validated, no regressions confirmed, commit message compliant

---

## Notes

- [P] tasks touch different files with no shared incomplete dependencies — safe to run in parallel
- Each user story has a clear, independently runnable validation step (see quickstart.md)
- T006 (Docker build) is a hard gate — do not proceed to Phase 4 without a clean Docker build
- Constitution principle II (overhead) is enforced by T015 — this must pass before the change is considered complete
- Constitution commit-message rule is enforced by T018 — include "OMR"/"BMR" in the eviction-logic commit
