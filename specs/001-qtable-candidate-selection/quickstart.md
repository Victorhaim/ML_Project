# Quickstart Validation Guide: TLCacheQL

## Prerequisites

- Docker installed and running (Docker Desktop on Mac — no `sudo` needed).
- Repository root contains `3LCache/`, `3LCacheQL/`, `libCacheSim/`, and `dockerfile`.
- Trace files in tab-separated `time id size` format in `data/`.

## Step 1: Build the Docker image

From the repository root:

```bash
docker build -t 3lcache -f dockerfile .
docker run -v /Users/victor_haim/Documents/technion/AI_PROJECT/ML_Project/data:/data -it 3lcache bash
```

The binary is pre-built inside the image at `/build/_build/bin/cachesim`. No rebuild needed inside the container.

**Expected**: shell prompt inside container, `ls /build/_build/bin/cachesim` shows the binary.

## Step 2: Smoke-test the new variant (SC-001)

All commands run **inside the container**. Trace files are available under `/data/`.

General form:

```bash
/build/_build/bin/cachesim \
  <trace_path> csv <eviction_algo> <cache_size_bytes> \
  -t "time-col=1,obj-id-col=2,obj-size-col=3,obj-id-is-num=1"
```

Example — 3LCacheQL on w43.csv with 100 MB cache:

```bash
/build/_build/bin/cachesim /data/w43.csv csv 3lcacheql 104857600 \
  -t "time-col=1,obj-id-col=2,obj-size-col=3,obj-id-is-num=1"
```

Baseline LRU for comparison:

```bash
/build/_build/bin/cachesim /data/w43.csv csv lru 104857600 \
  -t "time-col=1,obj-id-col=2,obj-size-col=3,obj-id-is-num=1"
```

**Notes**:
- Column indices are 1-based.
- `obj-id-is-num=1` is required because object IDs in these traces are integers.
- The 10 INFO/DEBUG lines at startup are normal; the result line appears after processing all requests (wait ~30–60 seconds).

**Expected output** (stdout, after the debug lines):

```
/data/w43.csv TLCacheQL-BMR cache size   100MiB,    1000001 req, miss ratio 0.XXXX, throughput X.XX MQPS
```

**Verify**:
- Run completes without crash or assertion failure.
- `miss ratio` is in `[0, 1]`.
- 3LCacheQL miss ratio is lower than LRU miss ratio on the same trace.

## Step 3: Verify Q-table learning (SC-006)

Run a trace with ≥ 1M requests and check stderr for Q-table update counts:

```bash
/build/_build/bin/cachesim /data/w43.csv csv 3lcacheql 104857600 \
  -t "time-col=1,obj-id-col=2,obj-size-col=3,obj-id-is-num=1" \
  2>&1 | grep q_table_updates
```

**Expected**: lines matching `q_table_updates=<N>` with N non-zero and increasing.

## Step 4: Miss-ratio comparison (SC-002, SC-003)

From inside the container, with Python 3 available:

```bash
cd /build/3LCache/scripts
python3 miss_ratio_boxplot.py \
  --algo="['3lcacheql', '3lcache', 'lru', 'arc', 'tinylfu', 's3fifo', 'lecar', 'lhd', 'sieve', 'cacheus', 'gdsf']" \
  --dataset_path="/data/" \
  --dataset_info="./trace_info/dataset_info.txt" \
  --metric="bmr"
```

Notes:
- `--dataset_path` **must** end with `/` (no trailing slash causes path concatenation errors).
- The existing `dataset_info.txt` already covers the available traces; the script ignores entries whose files are absent from `/data/`.
- Results are written to `./result/<tracename>` by cachesim; box plots to `./figures/`.
- With 36 traces × 11 algorithms this run takes 20–40 minutes. Progress is silent; the plot is saved when done.

**Expected**: two PDF box plots in `scripts/figures/` (`bmr_for_small_cache_size.pdf`, `bmr_for_large_cache_size.pdf`); result files in `scripts/result/`.

Repeat with `--metric="omr"` for object miss ratio.

**Verify SC-003**: Inspect result files in `result/` — each should contain rows for both `TLCache-BMR` and `TLCacheQL-BMR` for direct per-trace comparison.

## Step 5: No-regression check (SC-004)

```bash
cd /build/3LCache/scripts
python3 miss_ratio_boxplot.py \
  --algo="['3lcache', 'lru']" \
  --dataset_path="/data/" \
  --dataset_info="./trace_info/dataset_info.txt" \
  --metric="bmr"
```

**Expected**: `3lcache` BMR values are identical to a baseline run made before the change. Record and compare.

## Step 6: CPU overhead check (SC-005)

```bash
cd /build/3LCache/scripts
python3 cpu_overhead_boxplot.py \
  --algo="['3lcacheql', '3lcache', 'lru']" \
  --dataset_path="/data/" \
  --dataset_info="./trace_info/dataset_info.txt"
```

**Expected**: `3lcacheql` overhead box does not exceed the `3lcache` threshold reported in the paper.

## Acceptance sign-off checklist

- [ ] Step 2: 3LCacheQL run completes, miss ratio reported, no crash (SC-001)
- [ ] Step 2: 3LCacheQL miss ratio ≤ LRU miss ratio on same trace
- [ ] Step 3: `q_table_updates` is non-zero and grows over a long trace (SC-006)
- [ ] Step 4: comparison figures and result files include `3lcacheql` (SC-002)
- [ ] Step 4: per-trace BMR columns for both `3lcache` and `3lcacheql` (SC-003)
- [ ] Step 5: `3lcache` BMR unchanged vs baseline (SC-004)
- [ ] Step 6: `3lcacheql` CPU overhead within `3lcache` threshold (SC-005)
