# Feature Specification: Regret-Based Expert Arbitration for 3L-Cache Eviction (3LCache+++)

**Feature Branch**: `002-rl-eviction-arbitration`

**Created**: 2026-07-06

**Status**: Draft

**Input**: User description: "the current change for the 3LCache (3LCache++) is not good and even do a worse performance than what we already have, I want to try a different method for improvement of the 3LCache algorithm, maybe with another RL method, in picking the candidates or in deciding who to evict — write a speckit of the best improvement we can add (so there will be 3LCacheNew or 3LCache+++)"

## Overview

3L-Cache evicts in two stages: a heuristic **candidate-selection** stage that samples a small set of eviction candidates from the cache queue, and a learned **victim-selection** stage that ranks those candidates by predicted reuse time and evicts the worst one.

The previous improvement attempt (the Q-table variant, feature 001 / "3LCache++") replaced the candidate-selection stage with a tabular reinforcement-learning policy. Its measured miss ratios were equal to or worse than baseline 3L-Cache. A post-mortem identifies three causes this feature must avoid:

1. **Unprotected exploration** — random exploration decisions directly evicted objects, and every wrong exploratory eviction costs real misses.
2. **State aliasing** — a very coarse discretized state treated dissimilar objects identically, so the learned values were noisy.
3. **No safety net** — the well-tuned heuristic sampler was discarded entirely; when the learned policy was wrong there was nothing to fall back to.

This feature introduces a **new variant, "3LCache+++"**, that keeps *both* existing stages intact and instead applies reinforcement learning to the **final eviction decision**: a small set of complementary **victim-ranking experts** each nominate a victim from the *same* candidate set, and a **regret-minimizing arbitration policy** learns online which expert to trust, per decision, from observed eviction outcomes. The baseline's reuse-time ranking is itself one of the experts, so the arbitration policy's worst case converges toward baseline 3L-Cache behavior rather than below it — directly addressing the failure mode of the previous attempt. This is the same family of online learning that LeCaR/CacheUS use to arbitrate between LRU and LFU, applied here one level up: arbitrating between 3L-Cache's learned ranking and complementary cheap rankings.

The candidate-selection stage (heuristic sampler and quick-demotion) is inherited **unchanged** — it is one of 3L-Cache's validated contributions and the previous attempt showed that replacing it is harmful.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run the new variant in a miss-ratio experiment (Priority: P1)

A cache researcher selects the new 3LCache+++ variant as the eviction policy in a standard eviction-simulation experiment, on a standard trace and cache size, and obtains object miss ratio (OMR) and byte miss ratio (BMR) results comparable against existing algorithms.

**Why this priority**: Without the variant being selectable and producing valid miss-ratio output, nothing else can be evaluated. This is the minimum viable product.

**Independent Test**: Select the new variant by name in a single-trace, single-cache-size simulation run and confirm the run completes and reports OMR and BMR without crashing or violating the cache-size constraint.

**Acceptance Scenarios**:

1. **Given** a standard trace and a configured cache size, **When** the experiment is run with the new variant selected as the eviction policy, **Then** the run completes and reports both OMR and BMR for that trace and size.
2. **Given** the new variant is run on the same trace and size as baseline 3L-Cache, **When** results are collected, **Then** the new variant's results appear alongside the baselines in the same result format.
3. **Given** a cache size smaller than the trace's working set, **When** the variant runs, **Then** the cached bytes never exceed the configured cache size at any point.

---

### User Story 2 - Eviction decided by learned expert arbitration (Priority: P1)

When an eviction is required, the variant gathers candidates exactly as baseline 3L-Cache does, obtains a victim nomination from each ranking expert over that candidate set, and evicts the victim nominated by the expert chosen by the arbitration policy. When an evicted object is re-requested soon after eviction, the arbitration policy is penalized for having trusted the expert responsible, and shifts trust toward experts that would have kept that object.

**Why this priority**: This is the substantive behavioral change that defines the feature. Without it, the variant is a clone of 3L-Cache.

**Independent Test**: Run the variant on a trace long enough for evictions and post-eviction re-requests to occur, and confirm via run statistics that (a) each eviction is attributed to a specific expert, and (b) the arbitration policy's trust distribution over experts changes during the run in response to eviction outcomes.

**Acceptance Scenarios**:

1. **Given** the cache is full and the victim-selection model is trained, **When** an eviction is triggered, **Then** the evicted object is the victim nominated by the expert selected by the arbitration policy for that decision.
2. **Given** an object evicted on the advice of one expert is re-requested while still inside the out-of-cache history window, **When** that re-request arrives, **Then** the arbitration policy records a regret event against that expert and decreases its future selection likelihood.
3. **Given** repeated regret events against one expert during a phase of the trace, **When** subsequent evictions occur in that phase, **Then** the arbitration policy selects that expert less often and the other experts more often.
4. **Given** the arbitration policy has driven an expert's trust very low, **When** the workload shifts so that expert becomes accurate again, **Then** the expert can recover selection likelihood (no expert is permanently eliminated).

---

### User Story 3 - Safety: never meaningfully worse than baseline 3L-Cache (Priority: P1)

A researcher compares the new variant head-to-head against baseline 3L-Cache across the standard dataset and confirms the new variant does not regress miss ratio beyond a small tolerance on any trace — the property the previous Q-table attempt lacked.

**Why this priority**: The explicit motivation for this feature is that the previous attempt *hurt* performance. A learned improvement that can degrade below baseline is not acceptable; the design's core promise is a bounded downside.

**Independent Test**: Run baseline 3L-Cache and the new variant on every trace of one standard dataset at the standard cache sizes and compare per-trace OMR and BMR deltas against the tolerance.

**Acceptance Scenarios**:

1. **Given** per-trace results for both the new variant and baseline 3L-Cache, **When** deltas are computed, **Then** no trace shows the new variant worse than baseline by more than the agreed tolerance (see SC-002).
2. **Given** a trace where the baseline's reuse-time ranking is consistently the best expert, **When** the run progresses, **Then** the arbitration policy converges to selecting the baseline expert for the large majority of evictions, and results track baseline results.

---

### User Story 4 - Compare against baselines and prior variants (Priority: P2)

A researcher runs the standard miss-ratio comparison across the new variant, baseline 3L-Cache, the previous Q-table variant, and the established baseline algorithms on at least one standard dataset, and reviews box plots comparing OMR and BMR.

**Why this priority**: Required to judge whether the new arbitration approach is the improvement the project is looking for, and to document that it beats the failed prior attempt; depends on P1 stories working first.

**Independent Test**: Run the standard miss-ratio comparison with the new variant added to the algorithm list and confirm a comparison figure and result files are produced that include the new variant.

**Acceptance Scenarios**:

1. **Given** the algorithm list includes the new variant, baseline 3L-Cache, the previous Q-table variant, and the established baselines, **When** the miss-ratio comparison is executed, **Then** box plots and result files are generated that include the new variant's OMR and BMR.
2. **Given** results for the new variant and the previous Q-table variant on the same traces, **When** they are compared, **Then** the new variant's per-trace OMR and BMR are reported side by side and the median result is at least as good.

---

### Edge Cases

- **Cold start (victim-selection model not yet trained)**: Baseline 3L-Cache uses a recency (LRU) fallback before its predictor is trained; the variant inherits this path unchanged, and expert arbitration only activates once candidate ranking is available.
- **All experts nominate the same victim**: The eviction proceeds normally; the regret event (if any) must not unfairly single out the selected expert when every expert endorsed the same victim.
- **Trust collapse**: An expert's trust must be bounded below by a floor so no expert's selection probability reaches zero permanently; workloads shift and eliminated experts cannot recover otherwise.
- **Evicted object never re-requested**: When an evicted object ages out of the out-of-cache history window without being re-requested, the pending attribution for it is resolved as a no-regret (or positive) outcome and its bookkeeping is released — attribution records must not accumulate without bound.
- **Very small cache (1–2 objects)**: The candidate set may contain a single object; arbitration must still yield a valid eviction and never select an object not in the cache.
- **Empty candidate situation**: If candidate gathering yields no candidate, the variant must fall back to a guaranteed eviction (inherited recency default) so admission can complete.
- **Multiple evictions per admission**: When several objects must be evicted to fit one new object, each eviction is an independent arbitration decision with its own attribution.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a new cache variant ("3LCache+++") that inherits behavior from baseline 3L-Cache and reuses its request handling, admission, queue bookkeeping, candidate-selection (sampling and quick-demotion), and miss-ratio accounting unchanged.
- **FR-002**: The new variant MUST be selectable as an eviction policy by a distinct name in the existing experiment workflow, independently of and without altering baseline 3L-Cache, the previous Q-table variant, or any other existing algorithm.
- **FR-003**: The variant MUST maintain at least two complementary victim-ranking experts that each nominate one victim from the *same* candidate set per eviction decision, one of which MUST be the baseline reuse-time ranking (so baseline behavior is always available as an expert). Additional experts MUST be cheap, deterministic rankings over already-tracked object attributes (e.g., recency-based and size-aware rankings).
- **FR-004**: For each eviction decision, the variant MUST select exactly one expert according to a learned trust distribution and evict that expert's nominated victim.
- **FR-005**: The trust distribution MUST update online from observed eviction outcomes: when an evicted object is re-requested within the existing out-of-cache history window, the expert(s) that nominated it MUST be penalized, with the penalty weighted so that quickly re-requested objects incur more regret than ones re-requested near the window's end; when an evicted object ages out of the window without re-request, the pending attribution MUST be resolved without penalty.
- **FR-006**: The variant MUST NOT include any exploration mechanism that evicts objects chosen at random or outside all experts' nominations; learning is confined to arbitration among expert nominations. (Lesson from the failed prior attempt: unprotected exploration in the eviction path directly costs misses.)
- **FR-007**: Each expert's selection probability MUST be bounded below by a configurable floor greater than zero, so no expert can be permanently starved and the policy can re-adapt when the workload shifts.
- **FR-008**: Before the victim-selection model is trained (cold start), the variant MUST behave identically to baseline 3L-Cache's untrained path (recency-based eviction).
- **FR-009**: When an admission requires space, the variant MUST repeatedly evict arbitration-chosen victims until the new object fits, and cached content MUST never exceed the configured cache size after an eviction cycle completes.
- **FR-010**: The variant MUST report object miss ratio (OMR) and byte miss ratio (BMR) in the same result format as existing algorithms and be runnable in the standard miss-ratio comparison alongside baseline 3L-Cache, the previous Q-table variant, and the established baselines, contributing to the generated figures and result files.
- **FR-011**: The variant MUST preserve the lookup / admit / evict interface contract so it integrates with the existing evaluation harness without changes to that harness's calling conventions.
- **FR-012**: Adding the variant MUST NOT change the behavior or results of baseline 3L-Cache or any other existing algorithm.
- **FR-013**: The variant MUST expose per-run observability sufficient to verify learning: per-expert selection counts, per-expert regret-event counts, and the trust distribution at periodic intervals.
- **FR-014**: The per-eviction cost added by arbitration (expert nominations over an already-gathered candidate set, one trust-distribution draw, and attribution bookkeeping) MUST be bounded by a constant factor of the candidate-set size, and pending attribution records MUST be bounded by the out-of-cache history window so memory does not grow with trace length.

### Key Entities *(include if feature involves data)*

- **Victim-Ranking Expert**: A deterministic rule that, given the eviction candidate set, nominates one victim. The baseline reuse-time ranking is always one expert; others rank by cheap, already-tracked attributes (recency, size).
- **Trust Distribution**: The learned per-expert weights that determine, per eviction decision, which expert's nomination is followed. Updated online from regret events; every weight is floored above zero.
- **Regret Event**: The outcome signal — an evicted object re-requested within the out-of-cache history window — attributed back to the expert(s) that nominated it, weighted by how soon the re-request occurred.
- **Eviction Attribution Record**: Pending bookkeeping linking an evicted object to the expert(s) that nominated it, held until the object is re-requested (regret) or ages out of the history window (no regret), then released.
- **Candidate Set**: The set of eviction candidates produced per eviction cycle by the inherited, unchanged 3L-Cache sampling stage.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The new variant runs to completion on every trace of at least one standard dataset at the standard cache sizes, producing both OMR and BMR, in 100% of attempted runs without crashes or cache-size violations.
- **SC-002** (safety): On every trace of the evaluated dataset, the new variant's OMR and BMR are each no worse than baseline 3L-Cache's by more than 1% relative; the previous attempt failed exactly this bar.
- **SC-003** (improvement): Across the evaluated dataset, the new variant's median BMR (for the byte-miss objective) and median OMR (for the object-miss objective) are equal to or better than baseline 3L-Cache's, with strict improvement on at least a quarter of the traces.
- **SC-004**: On every trace evaluated, the new variant's median OMR and BMR are at least as good as the previous Q-table variant's ("3LCache++"), demonstrating the new approach supersedes the failed attempt.
- **SC-005**: Adding the variant produces identical results for baseline 3L-Cache and all other existing algorithms compared to before the change (no regression in existing results).
- **SC-006**: The variant's per-request processing overhead remains within the project's documented low-overhead threshold relative to LRU, as measured by the standard CPU-overhead evaluation.
- **SC-007**: Learning is demonstrable from run statistics: on at least one standard trace, the trust distribution shifts measurably over the run (expert selection shares change by more than the floor-induced minimum), and per-expert regret counts are reported.

## Assumptions

- The variant is a new sibling algorithm registered with the evaluation harness under a distinct name (working name "3LCache+++"/"3LCacheNew"); it does not modify or remove baseline 3L-Cache, the previous Q-table variant, or any baseline algorithm. The previous Q-table variant is retained solely as a comparison point.
- "Best improvement" was delegated to the specification author: regret-based expert arbitration at the victim-selection step was chosen over (a) another candidate-selection replacement — because the prior attempt showed replacing the tuned sampler is harmful — and (b) RL-tuned sampler hyperparameters — because the sampler already self-adapts and offers less headroom. The decisive property is bounded downside: with baseline ranking as an expert, worst-case behavior converges toward baseline rather than below it.
- The arbitration policy is learned online within a single simulation run; there is no offline training phase and no persisted policy state across runs.
- Two to three experts are sufficient for v1; the expert set is fixed at run start (no dynamic expert addition/removal).
- The existing out-of-cache history window (the mechanism 3L-Cache already maintains for departed objects) is the observation channel for eviction outcomes; no new history structure is required.
- The 1% relative tolerance in SC-002 and the "quarter of traces" bar in SC-003 are working thresholds; they may be tightened or relaxed once first results exist, but any change must be recorded before final evaluation.
- Standard trace format and the existing dataset-info / cache-size configuration mechanism are reused unchanged; evaluation runs through the existing experiment scripts in the canonical Docker build environment, per the project constitution.
- "At least one standard dataset" is sufficient for v1 evaluation, consistent with the project's evaluation standard (Constitution Principle III).
