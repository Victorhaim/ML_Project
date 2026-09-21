#!/bin/bash
# Microsoft-only feature screen, self-contained.
#
# This one file owns the whole experiment: the 46-mask catalog, the persistent
# 80/20 MSR split, the TRAIN pass, the held-out TEST pass, and the result
# table. It calls cachesim directly and does not source any other script.
#
# Every mask gets its own policy table, profiler config and logs. The only
# thing shared across masks is the split manifest, so the feature mask is the
# single experimental factor.
#
#   ./run_msr_feature_screen.sh                    # all 46 sets
#   FEATURE_SETS="00 23 30" ./run_msr_feature_screen.sh
#   SPLIT_ONLY=1 ./run_msr_feature_screen.sh       # write split + catalog only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/external_data}"
MSR_SOURCE="${MSR_SOURCE:-msr_full}"
SCREEN_ROOT="${SCREEN_ROOT:-$SCRIPT_DIR/feature-screen/msr}"
CACHESIM="${CACHESIM:-$SCRIPT_DIR/_build/bin/cachesim}"
CACHE_SIZE="${CACHE_SIZE:-32MB}"
BASE_MAB_PARAMS="${BASE_MAB_PARAMS:-arm-count=6,arm-strategy=mode,mab-gamma=0.05,objective=byte-miss-ratio,arm-selector=knn}"
FEATURE_SETS="${FEATURE_SETS:-}"
RESET_SPLIT="${RESET_SPLIT:-0}"
RESET_SETS="${RESET_SETS:-0}"
SPLIT_ONLY="${SPLIT_ONLY:-0}"
# The small *_stats.csv are always written. KEEP_DETAIL_LOGS=1 additionally
# writes the per-regime dumps (regime_events / feature_samples), which are the
# largest files the run produces and are only a debugging aid.
KEEP_DETAIL_LOGS="${KEEP_DETAIL_LOGS:-0}"
KEEP_LINES="${KEEP_LINES:-TLCACHE_MAB_SUMMARY|[Ee][Rr][Rr][Oo][Rr]|terminate called|Assertion|Segmentation|std::bad|exception}"

SOURCE_DIR="$DATA_ROOT/$MSR_SOURCE"
SPLIT_DIR="$SCREEN_ROOT/split"
TRAIN_LIST="$SPLIT_DIR/train.txt"
TEST_LIST="$SPLIT_DIR/test.txt"
SPLIT_INFO="$SPLIT_DIR/split.info"
CATALOG_OUT="$SCREEN_ROOT/feature_sets.tsv"
RESULTS_CSV="$SCREEN_ROOT/screen_results.csv"

FEATURE_NAMES=(
    write_ratio avg_request_size size_variance singleton_ratio
    popularity_skewness avg_frequency object_diversity avg_reuse_distance
    sequentiality_ratio out_cache_hit_rate request_rate burstiness_index
    arrival_time_variance working_set_byte_delta scan_ratio
)

# ---------------------------------------------------------------- catalog ---
# id|kind|15-bit mask|active count|description
# 00-23 orthogonal screen (Plackett-Burman on 8 knowledge groups + foldover),
# 30 the all-15 anchor every other set is compared against, 31-38 whole-group
# drops, and the probes that test whether a group may stay a single switch.
feature_catalog() {
cat <<'CATALOG'
00|screen|111111100011100|10|write+size+pop+freq+time
01|screen|100110010000000|4|write+pop+reuse
02|screen|011000001000001|4|size+stream
03|screen|000111100100010|6|pop+freq+cache
04|screen|000000010011100|4|reuse+time
05|screen|100001101000001|5|write+freq+stream
06|screen|011001110100010|7|size+freq+reuse+cache
07|screen|011111111011101|12|size+pop+freq+reuse+stream+time
08|screen|111110011100011|10|write+size+pop+reuse+stream+cache
09|screen|000110001111111|9|pop+stream+cache+time
10|screen|111000000111110|8|write+size+cache+time
11|screen|100001111111111|11|write+freq+reuse+stream+cache+time
12|screen|000000011100011|5|reuse+stream+cache
13|screen|011001101111111|11|size+freq+stream+cache+time
14|screen|100111110111110|11|write+pop+freq+reuse+cache+time
15|screen|111000011011101|9|write+size+reuse+stream+time
16|screen|111111101100011|11|write+size+pop+freq+stream+cache
17|screen|011110010111110|10|size+pop+reuse+cache+time
18|screen|100110001011101|8|write+pop+stream+time
19|screen|100000000100010|3|write+cache
20|screen|000001100011100|5|freq+time
21|screen|111001110000000|6|write+size+freq+reuse
22|screen|000111111000001|7|pop+freq+reuse+stream
23|screen|011110000000000|4|size+pop
24|probe|110111111111111|14|all15-minus-size_variance
25|probe|111101111111111|14|all15-minus-popularity_skewness
26|probe|111111011111111|14|all15-minus-object_diversity
27|probe|111111111111110|14|all15-minus-scan_ratio
28|probe|111111111111101|14|all15-minus-working_set_byte_delta
29|probe|111111111110111|14|all15-minus-burstiness_index
30|anchor|111111111111111|15|all-15-comparison
31|loo|011111111111111|14|all15-minus-F_write
32|loo|100111111111111|13|all15-minus-F_size
33|loo|111001111111111|13|all15-minus-F_pop
34|loo|111110011111111|13|all15-minus-F_freq
35|loo|111111101111111|14|all15-minus-F_reuse
36|loo|111111110111110|13|all15-minus-F_stream
37|loo|111111111011101|13|all15-minus-F_cache
38|loo|111111111100011|12|all15-minus-F_time
39|probe|101111111111111|14|all15-minus-avg_request_size
40|probe|111011111111111|14|all15-minus-singleton_ratio
41|probe|111110111111111|14|all15-minus-avg_frequency
42|probe|111111110111111|14|all15-minus-sequentiality_ratio
43|probe|111111111011111|14|all15-minus-out_cache_hit_rate
44|probe|111111111101111|14|all15-minus-request_rate
45|probe|111111111111011|14|all15-minus-arrival_time_variance
CATALOG
}

validate_catalog() {
    local id kind mask expected note actual rows=0 seen=" "
    while IFS='|' read -r id kind mask expected note; do
        rows=$((rows + 1))
        [[ "$id" =~ ^[0-9][0-9]$ ]] ||
            { echo "ERROR: invalid set id: $id" >&2; return 1; }
        [[ "$mask" =~ ^[01]{15}$ ]] ||
            { echo "ERROR: set $id has a malformed mask: $mask" >&2; return 1; }
        actual="${mask//0/}"; actual="${#actual}"
        [ "$actual" -eq "$expected" ] ||
            { echo "ERROR: set $id claims $expected active, has $actual" >&2; return 1; }
        # A one- or two-dimensional KNN table is not a feature result.
        [ "$actual" -ge 3 ] ||
            { echo "ERROR: set $id enables fewer than 3 features" >&2; return 1; }
        [[ "$seen" != *" $mask "* ]] ||
            { echo "ERROR: set $id duplicates an earlier mask" >&2; return 1; }
        seen+="$mask "
    done < <(feature_catalog)
    [ "$rows" -eq 46 ] ||
        { echo "ERROR: expected 46 sets, found $rows" >&2; return 1; }
}

active_feature_names() {
    local mask="$1" i first=1
    for i in "${!FEATURE_NAMES[@]}"; do
        [ "${mask:i:1}" = 1 ] || continue
        [ "$first" = 1 ] || printf ','
        printf '%s' "${FEATURE_NAMES[$i]}"
        first=0
    done
    printf '\n'
}

is_selected() {
    local id="$1" want
    [ -z "$FEATURE_SETS" ] && return 0
    for want in $FEATURE_SETS; do
        [ "$want" = "$id" ] && return 0
    done
    return 1
}

# ------------------------------------------------------------------ split ---
collect_traces() {
    local -n _out="$1"
    local f base
    _out=()
    shopt -s globstar nullglob
    for f in "$SOURCE_DIR"/**/*.oracleGeneral.zst \
             "$SOURCE_DIR"/**/*.oracleGeneral.bin \
             "$SOURCE_DIR"/**/*.oracleGeneral; do
        [ -f "$f" ] && _out+=("$f")
    done
    if [ ${#_out[@]} -eq 0 ]; then
        for f in "$SOURCE_DIR"/**/*.csv; do
            [ -f "$f" ] || continue
            base="$(basename "$f")"
            [[ "$base" == *".sample.csv"* ]] && continue
            _out+=("$f")
        done
    fi
    [ ${#_out[@]} -gt 0 ] &&
        mapfile -t _out < <(printf '%s\n' "${_out[@]}" | LC_ALL=C sort -u)
}

manifest_bytes() {
    local manifest="$1" f total=0
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        total=$((total + $(stat -c '%s' "$f")))
    done < "$manifest"
    printf '%s\n' "$total"
}

create_split() {
    local -a traces=() entries=() train=() test=()
    local f size total=0 target largest="" largest_size=-1
    local test_bytes=0 old_gap new_gap smallest="" smallest_size=-1

    collect_traces traces
    if [ ${#traces[@]} -lt 2 ]; then
        echo "ERROR: $MSR_SOURCE needs >= 2 trace files, found ${#traces[@]}" >&2
        return 1
    fi

    for f in "${traces[@]}"; do
        size="$(stat -c '%s' "$f")"
        total=$((total + size))
        if [ "$size" -gt "$largest_size" ]; then
            largest="$f"; largest_size="$size"
        fi
        entries+=("${size}"$'\t'"${f}")
    done
    target=$((total / 5))

    # The largest trace always trains. For the rest, take a whole file into
    # TEST only while that moves TEST bytes closer to 20%. Descending size
    # order makes the split deterministic across runs.
    mapfile -t entries < <(printf '%s\n' "${entries[@]}" |
                           LC_ALL=C sort -t $'\t' -k1,1nr -k2,2)
    train+=("$largest")
    for f in "${entries[@]}"; do
        size="${f%%$'\t'*}"; f="${f#*$'\t'}"
        [ "$f" = "$largest" ] && continue
        old_gap=$((target > test_bytes ? target - test_bytes : test_bytes - target))
        new_gap=$((target > test_bytes + size
                   ? target - test_bytes - size
                   : test_bytes + size - target))
        if [ "$new_gap" -le "$old_gap" ]; then
            test+=("$f"); test_bytes=$((test_bytes + size))
        else
            train+=("$f")
        fi
    done

    # A held-out experiment needs at least one TEST trace.
    if [ ${#test[@]} -eq 0 ]; then
        for f in "${traces[@]}"; do
            [ "$f" = "$largest" ] && continue
            size="$(stat -c '%s' "$f")"
            if [ "$smallest_size" -lt 0 ] || [ "$size" -lt "$smallest_size" ]; then
                smallest="$f"; smallest_size="$size"
            fi
        done
        test=("$smallest"); train=()
        for f in "${traces[@]}"; do
            [ "$f" != "$smallest" ] && train+=("$f")
        done
    fi

    printf '%s\n' "${train[@]}" | LC_ALL=C sort > "$TRAIN_LIST"
    printf '%s\n' "${test[@]}"  | LC_ALL=C sort > "$TEST_LIST"
}

# -------------------------------------------------------------- cachesim ---
trace_type() {
    case "$(basename "$1")" in
        *.oracleGeneral.zst|*.oracleGeneral.bin|*.oracleGeneral) echo oracleGeneral ;;
        *) echo csv ;;
    esac
}

# MSR CSV puts the object id and size in columns 5 and 6.
trace_params() {
    [ "$(trace_type "$1")" = oracleGeneral ] && { echo ""; return 0; }
    echo "time-col=1,obj-id-is-num=true,obj-id-col=5,obj-size-col=6"
}

run_cachesim() {
    local trace="$1" extra="$2" out_file="$3"
    local ttype params status
    ttype="$(trace_type "$trace")"
    params="$(trace_params "$trace")"
    rm -f "$(basename "$trace").cachesim" 2>/dev/null || true
    set +e
        if [ -n "$params" ]; then
            { "$CACHESIM" "$trace" "$ttype" 3lcache-mab "$CACHE_SIZE" \
                -t "$params" -e "$extra" 2>&1
              echo "CACHESIM_EXIT=$?"; } |
                grep -E "$KEEP_LINES|CACHESIM_EXIT=" > "$out_file" || true
        else
            { "$CACHESIM" "$trace" "$ttype" 3lcache-mab "$CACHE_SIZE" \
                -e "$extra" 2>&1
              echo "CACHESIM_EXIT=$?"; } |
                grep -E "$KEEP_LINES|CACHESIM_EXIT=" > "$out_file" || true
        fi
    set -e
    rm -f "$(basename "$trace").cachesim" 2>/dev/null || true
    status="$(sed -n 's/^CACHESIM_EXIT=//p' "$out_file" | tail -n1)"
    status="${status:-1}"
    return "$status"
}

scrub_set_logs() {
    local set_dir="$1"
    [ "$KEEP_DETAIL_LOGS" = 1 ] && return 0
    # Raw cachesim text and per-regime event files are the space hogs.
    # Keep policy, config, run.info, test_per_trace.csv, and the small *stats*.csv.
    rm -f "$set_dir"/logs/train/*.log "$set_dir"/logs/test/*.log
    rm -f "$set_dir"/logs/train/train/regime_events.csv \
          "$set_dir"/logs/train/train/feature_samples.csv \
          "$set_dir"/logs/test/test/match_events.csv
}

summary_field() {
    awk -v key="$2" '/TLCACHE_MAB_SUMMARY/ {
        for (i = 1; i <= NF; i++) {
            split($i, kv, "=")
            if (kv[1] == key) value = kv[2]
        }
    } END { print value }' "$1"
}

# ------------------------------------------------------------------- main ---
validate_catalog
if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: MSR source folder not found: $SOURCE_DIR" >&2
    exit 1
fi
mkdir -p "$SCREEN_ROOT" "$SPLIT_DIR"
{
    printf 'set_id\tkind\tfeature_mask\tactive_count\tdescription\n'
    feature_catalog | tr '|' '\t'
} > "$CATALOG_OUT"

[ "$RESET_SPLIT" = 1 ] && rm -f "$TRAIN_LIST" "$TEST_LIST" "$SPLIT_INFO"
if [ ! -s "$TRAIN_LIST" ] || [ ! -s "$TEST_LIST" ]; then
    create_split
fi

TRAIN_BYTES="$(manifest_bytes "$TRAIN_LIST")"
TEST_BYTES="$(manifest_bytes "$TEST_LIST")"
TOTAL_BYTES=$((TRAIN_BYTES + TEST_BYTES))
TRAIN_COUNT="$(wc -l < "$TRAIN_LIST")"
TEST_COUNT="$(wc -l < "$TEST_LIST")"
{
    echo "source=$MSR_SOURCE"
    echo "train_files=$TRAIN_COUNT"
    echo "test_files=$TEST_COUNT"
    echo "train_bytes=$TRAIN_BYTES"
    echo "test_bytes=$TEST_BYTES"
    awk -v t="$TRAIN_BYTES" -v e="$TEST_BYTES" -v a="$TOTAL_BYTES" 'BEGIN {
        printf "train_pct=%.3f\ntest_pct=%.3f\n",
               a ? 100*t/a : 0, a ? 100*e/a : 0
    }'
} > "$SPLIT_INFO"

echo "MSR feature screen"
echo "  split: $TRAIN_COUNT TRAIN files / $TEST_COUNT TEST files"
awk -v t="$TRAIN_BYTES" -v e="$TEST_BYTES" -v a="$TOTAL_BYTES" 'BEGIN {
    printf "  bytes: %.1f%% TRAIN / %.1f%% TEST\n",
           a ? 100*t/a : 0, a ? 100*e/a : 0
}'
echo "  catalog: $CATALOG_OUT"

if [ "$SPLIT_ONLY" = 1 ]; then
    echo "SPLIT_ONLY=1, stopping before any run."
    exit 0
fi
if [ ! -x "$CACHESIM" ]; then
    echo "ERROR: cachesim not executable: $CACHESIM" >&2
    exit 1
fi

if [ ! -s "$RESULTS_CSV" ]; then
    echo "set_id,kind,active_count,feature_mask,test_bmr,shadow_bmr,delta_bmr,test_regimes,description" \
        > "$RESULTS_CSV"
fi

while IFS='|' read -r SET_ID SET_KIND MASK ACTIVE_COUNT DESCRIPTION; do
    is_selected "$SET_ID" || continue

    SET_DIR="$SCREEN_ROOT/set_${SET_ID}"
    [ "$RESET_SETS" = 1 ] && rm -rf "$SET_DIR"
    POLICY="$SET_DIR/meta_policy_${CACHE_SIZE}.txt"
    CONFIG="$SET_DIR/profiler_config_${CACHE_SIZE}.txt"
    PROGRESS="$SET_DIR/train_progress.log"
    TRAIN_LOG_DIR="$SET_DIR/logs/train"
    TEST_LOG_DIR="$SET_DIR/logs/test"
    mkdir -p "$SET_DIR" "$TRAIN_LOG_DIR" "$TEST_LOG_DIR"
    touch "$PROGRESS"

    ACTIVE_NAMES="$(active_feature_names "$MASK")"
    {
        echo "set_id=$SET_ID"
        echo "kind=$SET_KIND"
        echo "feature_mask=$MASK"
        echo "active_count=$ACTIVE_COUNT"
        echo "active_features=$ACTIVE_NAMES"
        echo "description=$DESCRIPTION"
        echo "cache_size=$CACHE_SIZE"
        echo "train_manifest=$TRAIN_LIST"
        echo "test_manifest=$TEST_LIST"
    } > "$SET_DIR/run.info"

    echo "============================================================"
    echo "SET $SET_ID ($SET_KIND) mask=$MASK active=$ACTIVE_COUNT"
    echo "  $DESCRIPTION"
    echo "============================================================"

    # ---- TRAIN: build this mask's own KNN table --------------------------
    while IFS= read -r TRACE; do
        [ -z "$TRACE" ] && continue
        REL="${TRACE#"$DATA_ROOT"/}"
        if grep -Fqx "TRAIN_DONE|$REL" "$PROGRESS"; then
            echo "  [train done] $REL"
            continue
        fi
        echo "  [train] $REL"
        OUT="$TRAIN_LOG_DIR/$(basename "$TRACE").log"
        if ! run_cachesim "$TRACE" \
            "${BASE_MAB_PARAMS},feature-mask=${MASK},profiler-mode=train,policy-file=${POLICY},config-file=${CONFIG},diag-enable=0,train-log-enable=0,event-logs=${KEEP_DETAIL_LOGS},train-log-prefix=${TRAIN_LOG_DIR}/mab" \
            "$OUT"; then
            echo "ERROR: set $SET_ID TRAIN failed on $REL (see $OUT)" >&2
            exit 1
        fi
        [ "$KEEP_DETAIL_LOGS" = 1 ] || rm -f "$OUT"
        echo "TRAIN_DONE|$REL" >> "$PROGRESS"
    done < "$TRAIN_LIST"

    if [ ! -s "$POLICY" ] || [ ! -s "$CONFIG" ]; then
        echo "ERROR: set $SET_ID produced no policy/config; refusing to TEST" >&2
        exit 1
    fi

    # ---- TEST: frozen table on the held-out traces -----------------------
    SET_TEST_CSV="$SET_DIR/test_per_trace.csv"
    echo "trace,bytes,mab_bmr,shadow_bmr,diff" > "$SET_TEST_CSV"
    while IFS= read -r TRACE; do
        [ -z "$TRACE" ] && continue
        REL="${TRACE#"$DATA_ROOT"/}"
        echo "  [test] $REL"
        OUT="$TEST_LOG_DIR/$(basename "$TRACE").log"
        if ! run_cachesim "$TRACE" \
            "${BASE_MAB_PARAMS},feature-mask=${MASK},profiler-mode=test,policy-file=${POLICY},config-file=${CONFIG},diag-enable=0,train-log-enable=0,event-logs=${KEEP_DETAIL_LOGS},train-log-prefix=${TEST_LOG_DIR}/mab" \
            "$OUT"; then
            echo "ERROR: set $SET_ID TEST failed on $REL (see $OUT)" >&2
            exit 1
        fi
        BYTES="$(summary_field "$OUT" bytes)"
        MAB_BMR="$(summary_field "$OUT" mab_bmr)"
        SHADOW_BMR="$(summary_field "$OUT" shadow_bmr)"
        if [ -z "$BYTES" ] || [ -z "$MAB_BMR" ] || [ -z "$SHADOW_BMR" ]; then
            echo "ERROR: set $SET_ID could not parse summary for $REL ($OUT)" >&2
            exit 1
        fi
        awk -v t="$(basename "$TRACE")" -v b="$BYTES" -v m="$MAB_BMR" -v s="$SHADOW_BMR" \
            'BEGIN { printf "%s,%s,%s,%s,%.6f\n", t, b, m, s, m - s }' \
            >> "$SET_TEST_CSV"
        [ "$KEEP_DETAIL_LOGS" = 1 ] || rm -f "$OUT"
    done < "$TEST_LIST"

    # Score the mask on the whole held-out set, weighted by trace bytes, so
    # the number never depends on how the mask happened to cut regimes.
    read -r SET_MAB SET_SHADOW SET_DELTA < <(
        awk -F, 'NR > 1 {
            bytes += $2; mab += $3 * $2; shadow += $4 * $2
        } END {
            if (bytes > 0)
                printf "%.6f %.6f %.6f\n", mab/bytes, shadow/bytes,
                                           (mab - shadow)/bytes
            else print "NA NA NA"
        }' "$SET_TEST_CSV"
    )
    REGIMES=0
    [ -s "$TEST_LOG_DIR/test/match_events.csv" ] &&
        REGIMES="$(($(wc -l < "$TEST_LOG_DIR/test/match_events.csv") - 1))"

    # Keep one row per set: re-running a set replaces its previous row.
    if [ -s "$RESULTS_CSV" ]; then
        grep -v "^${SET_ID}," "$RESULTS_CSV" > "$RESULTS_CSV.tmp" || true
        mv "$RESULTS_CSV.tmp" "$RESULTS_CSV"
    fi
    echo "$SET_ID,$SET_KIND,$ACTIVE_COUNT,$MASK,$SET_MAB,$SET_SHADOW,$SET_DELTA,$REGIMES,$DESCRIPTION" \
        >> "$RESULTS_CSV"
    echo "  -> mab_bmr=$SET_MAB shadow_bmr=$SET_SHADOW delta=$SET_DELTA (negative is better)"
    scrub_set_logs "$SET_DIR"
done < <(feature_catalog)

echo ""
echo "============================================================"
echo "Feature screen complete."
echo "  results: $RESULTS_CSV"
echo "  per set: $SCREEN_ROOT/set_<id>/"
echo "  compare every set against set 30 (all 15 features)"
echo "============================================================"
{
    head -n1 "$RESULTS_CSV"
    tail -n +2 "$RESULTS_CSV" | LC_ALL=C sort -t, -k1,1
} | column -s, -t
