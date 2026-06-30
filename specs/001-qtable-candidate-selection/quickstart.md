# Quickstart Validation Guide: TLCacheQL

## Prerequisites

- Docker installed and running.
- Repository cloned with `3LCache/` and `libCacheSim/` subdirectories present.
- At least one trace file in space-separated `time id size` format available at `<dataset_path>`.
- `trace_info/dataset_info.txt` updated with the trace's unique-byte count.

## Step 1: Build inside Docker

```bash
sudo docker build -t 3lcache -f dockerfile .
sudo docker run -v /local/data:/data -it 3lcache bash
```

Inside the container:

```bash
cd /build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

**Expected**: build completes without errors; `_build/bin/cachesim` is present.

## Step 2: Smoke-test the new variant (SC-001)

Run a single-trace, single-size simulation:

```bash
./_build/bin/cachesim \
  <trace_path> \
  csv \
  -t "time-col=0,obj-id-col=1,obj-size-col=2" \
  -a 3lcacheql \
  -s <cache_size_bytes> \
  -e "objective=byte-miss-ratio"
```

**Expected output** (stdout):
```
TLCacheQL-BMR cache size <N>, ...
... object miss ratio: <OMR>, byte miss ratio: <BMR>
```

**Verify**:
- Run completes without crash or assertion failure.
- `<OMR>` and `<BMR>` are in `[0, 1]`.
- Re-run baseline: `... -a 3lcache ...` — result for `3lcache` is unchanged.

## Step 3: Verify Q-table learning (SC-006)

Run a longer trace (≥ 1M requests) and check that `q_table_updates` grows:

The `update_stat_periodic()` override logs `q_table_updates` to stderr. Confirm the value is non-zero and increasing over time by checking for a line matching `q_table_updates=<N>` in stderr output after the run.

## Step 4: Miss-ratio comparison (SC-002, SC-003)

Inside the container or with Python 3 available:

```bash
cd 3LCache/scripts
python3 miss_ratio_boxplot.py \
  --algo="['3lcacheql', '3lcache', 'lru', 'arc', 'tinylfu', 's3fifo', 'lecar', 'lhd', 'sieve', 'cacheus', 'gdsf']" \
  --dataset_path="<dataset_path>" \
  --dataset_info="./trace_info/dataset_info.txt" \
  --metric="bmr"
```

**Expected**: box plot PNG written to `scripts/figures/`; result CSV written to `scripts/result/`. Both contain a column/entry for `3lcacheql`.

Repeat with `--metric="omr"` for object miss ratio.

**Verify SC-003**: Inspect result CSV — each row has both `3lcache` and `3lcacheql` columns for direct per-trace comparison.

## Step 5: No-regression check (SC-004)

```bash
python3 miss_ratio_boxplot.py \
  --algo="['3lcache', 'lru']" \
  --dataset_path="<dataset_path>" \
  --dataset_info="./trace_info/dataset_info.txt" \
  --metric="bmr"
```

**Expected**: `3lcache` BMR values are identical to a baseline run made before the change. Record and compare.

## Step 6: CPU overhead check (SC-005)

```bash
python3 cpu_overhead_boxplot.py \
  --algo="['3lcacheql', '3lcache', 'lru']" \
  --dataset_path="<dataset_path>" \
  --dataset_info="./trace_info/dataset_info.txt"
```

**Expected**: `3lcacheql` overhead box does not exceed the `3lcache` threshold reported in the paper. The box plot in `scripts/figures/` should show `3lcacheql` within the same order of magnitude as `3lcache` relative to `lru`.

## Acceptance sign-off checklist

- [ ] Step 2: single-trace run completes, OMR and BMR reported, no crash (SC-001)
- [ ] Step 2: baseline `3lcache` results unchanged (SC-004)
- [ ] Step 3: `q_table_updates` grows from 0 over a long trace (SC-006)
- [ ] Step 4: comparison figures and result files include `3lcacheql` (SC-002)
- [ ] Step 4: per-trace OMR and BMR columns for both `3lcache` and `3lcacheql` (SC-003)
- [ ] Step 6: `3lcacheql` CPU overhead within `3lcache`'s threshold (SC-005)
