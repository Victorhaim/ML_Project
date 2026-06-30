<!--
  Sync Impact Report
  ==================
  Version change: 1.0.0 → 1.1.0
  Modified principles:
    - IV. Interface Compatibility — added Docker build gate alongside script gate
  Modified sections:
    - Development Workflow — Docker is now the REQUIRED build environment;
      direct host builds are disallowed
  Removed sections: None
  Templates requiring updates:
    - .specify/templates/plan-template.md  ✅ aligned — Constitution Check gate references these principles
    - .specify/templates/spec-template.md  ✅ aligned — no new mandatory sections required
    - .specify/templates/tasks-template.md ✅ aligned — setup tasks should reference Docker
  Deferred TODOs: None
-->

# 3L-Cache Constitution

## Core Principles

### I. Reproducibility First

Every experiment, benchmark, and result MUST be fully reproducible from a documented
command sequence. Scripts, trace paths, dataset info files, algorithm lists, and cache
sizes used in any published result MUST be committed to the repository. Results that
cannot be re-run from the repo's scripts are invalid for comparison or publication.

**Rationale**: This is a research artifact tied to a paper; reproducibility is the
primary evidence of correctness. Ad-hoc or undocumented runs cannot be peer-reviewed
or extended.

### II. Low Overhead Guarantee

The core eviction pipeline (lookup / admit / evict) MUST remain bounded in CPU
overhead relative to LRU. Any new feature or model update MUST be validated for CPU
overhead using `cpu_overhead_boxplot.py` before merge. Features that regress CPU
overhead beyond the paper's reported threshold MUST NOT be merged without explicit
justification and updated benchmarks.

**Rationale**: "Low overhead" is a foundational claim of 3L-Cache and appears in the
paper title. Losing this property invalidates the contribution.

### III. Evaluation-Driven Validation

All algorithmic changes to 3LCache, 3LCache+, or 3LCache++ MUST be validated by
running miss ratio comparisons against the baseline algorithms (LRU, ARC, TinyLFU,
S3-FIFO, LeCaR, LHD, SIEVE, CacheUS, GDSF) on at least one standard dataset before
the change is considered complete. Both object miss ratio (OMR) and byte miss ratio
(BMR) MUST be reported when changing eviction logic.

**Rationale**: Miss ratio improvement over baselines is the core empirical claim.
Unvalidated changes may silently regress the algorithm's effectiveness.

### IV. Interface Compatibility with libCacheSim

The `lookup()`, `admit()`, and `evict()` method signatures MUST remain compatible with
the libCacheSim interface contract. Any change that breaks this contract requires a
corresponding update to the interface adapter files (`TLCache_Interface.cpp`,
`TLCacheN_Interface.cpp`) and MUST be explicitly documented in the commit message.
The build MUST succeed inside Docker (via the project `dockerfile`) after any
structural change. Direct host builds are not a valid substitute for the Docker
build gate.

**Rationale**: 3L-Cache is built on libCacheSim; breaking the interface silently
breaks the entire evaluation harness and makes the artifact unusable. Docker
ensures the Ubuntu 18.04 + cmake 3.28.6 toolchain matches the canonical
environment specified in the README.

### V. Algorithm Simplicity

The core cache logic in `TLCache.cpp` / `TLCacheN.cpp` MUST remain readable and
minimal. Abstractions are only justified when they eliminate actual duplication across
the 3LCache / 3LCache+ / 3LCache++ variants. Model training logic and eviction
sampling MUST NOT be interleaved with cache bookkeeping in ways that obscure the
algorithm flow. Complexity MUST be justified in the PR description, not through
comments in the code.

**Rationale**: The algorithm must be understandable to reviewers and future researchers.
Premature abstractions in research code reduce clarity without benefit.

## Experiment & Evaluation Standards

- Trace files MUST be in space-separated CSV format with columns: `time` (int64),
  `id` (int64), `size` (uint32). Deviating formats MUST use the `-t` flag and
  document the mapping.
- Dataset info files (`trace_info/dataset_info.txt`) MUST be updated whenever a new
  trace is added, recording the number of unique bytes for that trace.
- Experiment results are written to `scripts/result/` and figures to `scripts/figures/`.
  These directories MUST NOT be committed; they are generated artifacts.
- Cache size parameters used in paper-reported experiments MUST be documented in
  `trace_info/` so experiments can be reproduced exactly.
- Multi-size runs (comma-separated cache sizes in `cachesim` invocations) MUST report
  results for each size individually.

## Development Workflow

- All builds MUST go through Docker using the project `dockerfile` at the repo root.
  The canonical environment is **Ubuntu 18.04 with cmake 3.28.6**. Direct host builds
  are not permitted as a final validation step.
  ```bash
  # Build the image
  sudo docker build -t 3lcache -f dockerfile .

  # Run inside the container (mount local data)
  sudo docker run -v /local/data:/data -it 3lcache bash
  ```
- The `_build/` directory is generated; it MUST NOT be committed.
- New variants (e.g., a hypothetical 3LCache+++) MUST follow the existing file naming
  convention (`TLCacheX.cpp`, `TLCacheX.h`, `TLCacheX_Interface.cpp`) and register
  with libCacheSim's algorithm registry.
- Python evaluation scripts MUST be compatible with Python 3 and accept
  `--dataset_path`, `--dataset_info`, `--algo`, and (where applicable) `--metric`
  flags consistent with existing scripts.
- Commits touching eviction logic MUST include the words "OMR" or "BMR" in the
  commit message summarizing the measured impact, or explicitly state "no miss ratio
  change" if the change is structural only.

## Governance

This constitution supersedes any informal conventions in the repository. Amendments
require:

1. A written rationale explaining why the principle must change.
2. An updated version number per the policy below.
3. A review of all template files to propagate the change.

**Versioning policy**:
- MAJOR bump: Removal or incompatible redefinition of an existing principle.
- MINOR bump: New principle or section added; material expansion of existing guidance.
- PATCH bump: Clarifications, wording fixes, non-semantic refinements.

**Compliance**: Every feature plan (plan.md) MUST include a "Constitution Check"
section verifying that the planned implementation satisfies all five core principles.
The check MUST be re-evaluated after the design phase before implementation begins.

**Version**: 1.1.0 | **Ratified**: 2026-06-30 | **Last Amended**: 2026-06-30
