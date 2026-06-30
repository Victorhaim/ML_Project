# Feature Specification: Q-Table Candidate Selection for 3L-Cache

**Feature Branch**: `001-qtable-candidate-selection`

**Created**: 2026-06-30

**Status**: Draft

**Input**: User description: "I want to add a new algorithm that inherits from 3LCache and replace the candidates selection policy with QTable instead of the algorithm we have now. The Q-table should not predict candidate reuse time but should sample the candidate to pick for eviction."

## Overview

3L-Cache currently selects eviction victims in two stages: a **heuristic candidate-selection (sampling) policy** that scans the cache queue and gathers a small set of eviction candidates (driven by frequency boundaries, scan length, reserved-space ratios, and an adaptive LRU-sampling fraction), followed by a learned reuse-time predictor that ranks those candidates.

This feature introduces a **new cache variant that inherits from 3L-Cache** and replaces *only* the heuristic candidate-selection policy with a **tabular reinforcement-learning (Q-table) policy**. The Q-table does **not** predict reuse time; it learns, from experience accumulated during the run, which objects to sample and pick as eviction candidates. All other behavior of 3L-Cache (request handling, admission, queue bookkeeping, miss-ratio accounting) is inherited unchanged.

The goal is to evaluate whether a learned (Q-table) candidate-selection policy improves miss ratio relative to the existing heuristic sampler, while preserving 3L-Cache's low-overhead and interface guarantees.

## Clarifications

### Session 2026-06-30

- Q: Does the inherited reuse-time predictor stay in the eviction path, or is it removed so the Q-table alone determines victims? → A: Keep it — the Q-table samples candidates, the inherited GBM reuse-time predictor ranks those candidates, and the worst-predicted-reuse object is evicted.
- Q: What outcome signal rewards/penalizes a candidate-selection decision? → A: Penalize evicting an object that is re-requested soon after eviction (detected via the existing out-of-cache history window); reward when it is not re-requested.
- Q: Which object/queue features form the Q-table's state? → A: Discretized buckets of (frequency, recency/age, size) — reusing the per-object attributes 3L-Cache already tracks.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run the new variant in a miss-ratio experiment (Priority: P1)

A cache researcher runs an eviction-simulation experiment selecting the new Q-table variant as the eviction policy, on a standard trace and cache size, and obtains object miss ratio (OMR) and byte miss ratio (BMR) results that can be compared against existing algorithms.

**Why this priority**: This is the core deliverable — without the variant being selectable and producing valid miss-ratio output, nothing else can be evaluated. It is the minimum viable product.

**Independent Test**: Select the new variant by name in a single-trace, single-cache-size simulation run and confirm the run completes and reports OMR and BMR without crashing or violating the cache-size constraint.

**Acceptance Scenarios**:

1. **Given** a standard trace and a configured cache size, **When** the experiment is run with the new variant selected as the eviction policy, **Then** the run completes and reports both OMR and BMR for that trace and size.
2. **Given** the new variant is run on the same trace and size as baseline 3L-Cache, **When** results are collected, **Then** the new variant's results appear alongside the baselines in the same result format.
3. **Given** a cache size smaller than the trace's working set, **When** the variant runs, **Then** the cached bytes never exceed the configured cache size at any point.

---

### User Story 2 - Candidate selection driven by the Q-table (Priority: P1)

When the cache is full and an eviction is required, the variant uses the Q-table policy (rather than the inherited heuristic sampler) to choose which cached object(s) become eviction candidates, and evicts accordingly. The Q-table updates from the outcomes of its choices as the run proceeds.

**Why this priority**: This is the substantive behavioral change that defines the feature. Without it, the variant is just a clone of 3L-Cache.

**Independent Test**: Run the variant on a trace long enough for the policy to accumulate experience, and confirm (via run statistics/logging) that eviction candidates are produced by the Q-table policy and that the policy's internal table changes over the run.

**Acceptance Scenarios**:

1. **Given** the cache is full and a new object must be admitted, **When** an eviction is triggered, **Then** the evicted object is one chosen by the Q-table candidate-selection policy.
2. **Given** the run has processed enough requests to gather experience, **When** the policy makes subsequent candidate-selection decisions, **Then** those decisions reflect updates learned from earlier eviction outcomes (the policy is not static).
3. **Given** the policy has not yet accumulated any experience (cold start), **When** an eviction is required, **Then** the variant still selects a valid candidate using a defined fallback (e.g., the inherited recency-based default) without failing.

---

### User Story 3 - Compare against baselines and the original 3L-Cache (Priority: P2)

A researcher runs the standard miss-ratio comparison across the new variant, baseline 3L-Cache, and the established baseline algorithms on at least one standard dataset, and reviews a box plot comparing OMR and BMR.

**Why this priority**: Required to judge whether the Q-table policy is an improvement, but depends on P1 working first.

**Independent Test**: Run the standard miss-ratio comparison with the new variant added to the algorithm list and confirm a comparison figure and result files are produced that include the new variant.

**Acceptance Scenarios**:

1. **Given** the algorithm list includes the new variant and the existing baselines, **When** the miss-ratio comparison is executed, **Then** a box plot and result files are generated that include the new variant's OMR and BMR.
2. **Given** results for the new variant and baseline 3L-Cache on the same traces, **When** they are compared, **Then** the difference in OMR and BMR per trace is reported and reviewable.

---

### Edge Cases

- **Cold start / no experience yet**: Before the policy has gathered enough experience, eviction must still proceed via a defined fallback selection rule.
- **Very small cache (1–2 objects)**: The policy must still produce a valid eviction victim and never select an object that is not in the cache.
- **Trace with highly skewed or uniform popularity**: The policy must not stall (must always be able to free enough space to admit a new object), continuing to evict until space suffices.
- **State never before seen**: When the policy encounters an object/queue state it has no recorded value for, it must select a valid candidate using a defined default rather than failing.
- **Empty candidate situation**: If the policy's selection yields no candidate, the variant must fall back to a guaranteed eviction so the admit operation can complete.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a new cache variant that inherits behavior from 3L-Cache and reuses 3L-Cache's request handling, admission, queue bookkeeping, and miss-ratio accounting unchanged.
- **FR-002**: The new variant MUST be selectable as an eviction policy by a distinct name in the existing experiment workflow, independently of and without altering baseline 3L-Cache.
- **FR-003**: The variant MUST replace 3L-Cache's heuristic candidate-selection (sampling) policy with a Q-table policy that selects which cached object(s) become eviction candidates.
- **FR-004**: The Q-table policy MUST select eviction candidates without producing a reuse-time prediction; candidate selection is the policy's only responsibility.
- **FR-005**: The Q-table MUST update online from the observed outcomes of its candidate-selection decisions, with no dependency on a separately trained offline model. The reward signal MUST penalize selecting (and thereby evicting) an object that is re-requested soon after eviction — detected via the existing out-of-cache history window — and reward selections of objects that are not re-requested within that window.
- **FR-006**: When the cache is full and an admission requires space, the variant MUST repeatedly evict objects chosen by the Q-table policy until the new object fits within the configured cache size.
- **FR-007**: The variant MUST define and apply a fallback candidate-selection rule for cold-start and unseen-state situations so eviction always succeeds.
- **FR-008**: The variant MUST report object miss ratio (OMR) and byte miss ratio (BMR) in the same result format as existing algorithms.
- **FR-009**: The variant MUST never allow cached content to exceed the configured cache size after an eviction cycle completes.
- **FR-010**: The variant MUST be runnable in the standard miss-ratio comparison alongside baseline 3L-Cache and the established baseline algorithms, contributing its results to the generated comparison figures and result files.
- **FR-011**: The variant MUST preserve the lookup / admit / evict interface contract so it integrates with the existing evaluation harness without changes to that harness's calling conventions.
- **FR-012**: Adding the variant MUST NOT change the behavior or results of baseline 3L-Cache or any other existing algorithm.
- **FR-013**: The eviction path MUST retain the inherited reuse-time predictor: the Q-table policy selects the eviction candidate set, the reuse-time predictor ranks that set, and the candidate with the worst predicted reuse is evicted. The Q-table changes *which objects are candidates*; the predictor still chooses the victim among them.
- **FR-014**: The Q-table's state MUST be formed from discretized buckets of an object's (frequency, recency/age, size), reusing the per-object attributes 3L-Cache already tracks. The number of buckets per dimension MUST be bounded so the total state space stays small enough not to dominate memory or per-request lookup cost.

### Key Entities *(include if feature involves data)*

- **Q-Table Policy**: The learned candidate-selection mechanism. Maps a discretized state of an object (and/or its queue context) to a learned value guiding whether it should be picked as an eviction candidate. Updated from eviction outcomes during the run.
- **State**: A bounded, discretized description of a cached object formed from buckets of its frequency, recency/age, and size. The set of possible states is finite and small (bounded buckets per dimension).
- **Action**: The selection decision the policy makes for a candidate (e.g., pick this object as an eviction candidate vs. skip it).
- **Reward / Outcome Signal**: The feedback derived from what happens after an eviction decision (e.g., whether an evicted object is soon re-requested), used to update the Q-table.
- **Eviction Candidate**: A cached object chosen by the policy as eligible for eviction during an eviction cycle.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The new variant can be selected and run to completion on at least one standard trace at a configured cache size, producing both OMR and BMR, in 100% of attempted runs without crashes or cache-size violations.
- **SC-002**: The variant appears in the standard miss-ratio comparison output (figure and result files) alongside baseline 3L-Cache and the established baselines on at least one standard dataset.
- **SC-003**: Per-trace OMR and BMR differences between the new variant and baseline 3L-Cache are reported and reviewable for every trace in the comparison.
- **SC-004**: Adding the variant produces identical results for baseline 3L-Cache and all other existing algorithms compared to before the change (no regression in existing results).
- **SC-005**: The variant's per-request processing overhead remains within 3L-Cache's documented low-overhead threshold relative to LRU, as measured by the standard CPU-overhead evaluation.
- **SC-006**: The Q-table policy demonstrably learns: candidate-selection decisions for a given recurring state change over the course of a run as outcomes accumulate (observable in run statistics).

## Assumptions

- The variant is a new sibling algorithm registered with the evaluation harness; it does not modify or remove baseline 3L-Cache.
- "Replace the candidate-selection policy" refers specifically to the heuristic sampling stage that gathers eviction candidates from the cache queue — not the request/admission path, which is inherited unchanged.
- The Q-table is learned **online**, within the same simulation run; there is no separate offline training phase or persisted model file required for v1.
- The state space is intentionally small and discretized (tabular Q-learning), so the Q-table itself does not become a significant memory or CPU cost.
- Standard trace format (space-separated `time`, `id`, `size`) and the existing dataset-info / cache-size configuration mechanism are reused unchanged.
- Evaluation is performed through the existing experiment scripts and the canonical Docker build environment; no new evaluation tooling is required beyond adding the variant to the algorithm list.
- A recency-based default (consistent with 3L-Cache's pre-training LRU fallback) is an acceptable cold-start / unseen-state fallback unless specified otherwise.
- "At least one standard dataset/trace" is sufficient to consider the evaluation complete for v1, consistent with the project's evaluation standard.
