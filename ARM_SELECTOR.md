# Arm selector training and TEST flow

Two LightGBMs, two jobs:

- **First LightGBM (this file):** at window **open**, pick a **starting** sampling mix (one of four corners). Replaces KNN / the policy table.
- **Second LightGBM (3L-Cache `prediction`):** ranks objects. Unchanged. Learns reuse distance, not arms.

Inside a window the **article-style slider** moves the four queue-band shares a little from eviction counts (`±0.05`, floor `0.05`). That is **not** saved as "the arm." The first tree never learns the ±1 steps.

Window open/close still uses **stream** feature drift (and byte shock) only. Cache occupancy does **not** close a regime. Cache fill / miss EMA / ghost hit rate **are** inputs to the first tree.

## 1. Collect labels and train

No Python. Collect writes the window CSV, then the same `cachesim` binary fits both first-tree LightGBMs (`arm-selector=fit`).

```bash
./train_arm_selector.sh
```

Each stream **cell** (behavior type × size area) keeps a shuffled **queue of the four start arms**. Enter a window → take the next unused start → hold that **start** for the whole window while the slider may move the mix. One CSV line per window:

`features at open + cache state + start arm → whole-window delta vs shadow`

Losses are kept. There is no mid-window arm swap and no freeze/replay of four caches.

Just `./train_arm_selector.sh`. Old leftover CSVs are wiped automatically; a current collect resumes. `RESET_ARM_COLLECTION=1` only if you want to force a wipe.

## 2. TEST

```bash
./run_experiments.sh
```

Until the profiler window is full, start with Explore (arm 3). When features are ready, and on every later `regime_reset`, the first LightGBM scores all four **starts** and picks one. Then only the slider and the eviction tree run until the next window.

```text
score = (1 - lambda) * predicted_mean_delta
      + lambda * predicted_25th_percentile_delta
```

`TLCACHE_MAB_SUMMARY` includes `arm_slider_steps=` (how many article-sized mix moves ran).

## Statistics logs (same matrices as before)

TRAIN collect writes under `cdt_logs/train/` (prefix is required; `train-log-enable` can stay 0):

| file | what |
|---|---|
| `regime_stats.csv` | size-area win/loss, median length, beat-baseline |
| `type_stats.csv` | behavior type × size area |
| `feature_stats.csv` | per-feature coverage / range |
| `regime_events.csv` | one row per closed window |
| `feature_samples.csv` | feature vector at each close |

TEST writes under `cdt_logs/test/`:

| file | what |
|---|---|
| `type_stats.csv` | same type × size matrix on held-out windows |
| `match_events.csv` | per window: type, bytes, delta, start arm / mix |
| `match_stats.csv` | KNN distance bands (legacy). Model runs land in `unmatched` |
| `arm_model_events.csv` | first-tree scores and realized delta |

## Start mixes (corners)

| arm | name | (Q1 newest … Q4 oldest) |
|---|---|---|
| 0 | Balanced | 0.50, 0.00, 0.00, 0.50 |
| 1 | Tail+ | 0.00, 0.00, 0.50, 0.50 |
| 2 | Head+ | 0.50, 0.50, 0.00, 0.00 |
| 3 | Explore | 0.25, 0.25, 0.25, 0.25 |

After a full queue walk, eviction/sample rates per band move `MIX_STEP` from the worst band to the best, never below `MIX_MIN_SHARE`.
