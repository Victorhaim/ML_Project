#!/bin/bash
# Online training: one in-process ThreeLCache shadow supplies the true regime
# delta and one coverage-preserving box table is written per cache size.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# TRAIN uses the FULL dataset behind the external_data symlink (all vendors:
# alibaba, twitter, meta, ...). TEST is the disjoint held-out set under ./data,
# so the split is by folder here, not by source filter. DATA_DIR overrides both.
if [ -n "${DATA_DIR:-}" ]; then
    :
elif [ -d "$SCRIPT_DIR/external_data" ]; then
    DATA_DIR="$SCRIPT_DIR/external_data"
elif [ -d "$SCRIPT_DIR/data" ]; then
    DATA_DIR="$SCRIPT_DIR/data"
else
    DATA_DIR="$SCRIPT_DIR/external_data"
fi
INTERMEDIATE_DIR="$SCRIPT_DIR/intermediate-results"
# Policy matrix + profiler config live in intermediate-results so TRAIN writes
# and TEST reads the same place. Override POLICY_DIR only to relocate both.
POLICY_DIR="${POLICY_DIR:-$INTERMEDIATE_DIR}"
LOG_OUTPUT_DIR="$SCRIPT_DIR/cdt_logs"
CACHESIM="${CACHESIM:-./_build/bin/cachesim}"

TRAIN_OUTPUT_DIR="$LOG_OUTPUT_DIR/train"
mkdir -p "$INTERMEDIATE_DIR" "$POLICY_DIR" "$LOG_OUTPUT_DIR" "$TRAIN_OUTPUT_DIR"

# CSV column mapping (ignored for oracleGeneral binary traces).
TRACE_PARAMS_COMMON="time-col=1,obj-id-is-num=true,obj-id-col=2,obj-size-col=3"
TRACE_PARAMS_MSR="time-col=1,obj-id-is-num=true,obj-id-col=5,obj-size-col=6"
# Optional vendor filters, space-separated path substrings.
# Example: TRAIN_SOURCES="alibaba microsoft meta"
TRAIN_SOURCES="${TRAIN_SOURCES:-}"

# Confirm run default: one size. Override e.g. CACHE_SIZES="16MB 32MB 64MB".
CACHE_SIZES="${CACHE_SIZES:-32MB}"
# Path A defaults (mode arms, stable Exp3).
MAB_PARAMS="${MAB_PARAMS:-arm-count=4,arm-strategy=mode,mab-gamma=0.05,objective=byte-miss-ratio}"
# Primary compare metric: bmr (byte) or omr (object/request).
MISS_METRIC="${MISS_METRIC:-bmr}"

# Held-out split is by data folder (external_data vs ./data), so TRAIN_SOURCES
# is optional. Set it only to narrow which external_data vendors train.

# Set FRESH_START=1 when intentionally retraining from zero. It removes only
# generated matrices/config/logs, never traces or source files.
FRESH_START="${FRESH_START:-0}"
# Normal runs resume from successful traces recorded in training_phase.log.
# Set RESUME_FROM_LOG=0 only when intentionally replaying completed traces
# into the existing policy.
RESUME_FROM_LOG="${RESUME_FROM_LOG:-1}"

# Resume hooks (empty = start from first trace).
START_16MB="${START_16MB:-}"
START_32MB="${START_32MB:-}"
START_64MB="${START_64MB:-}"

is_msr_trace() {
    local trace="$1"
    local base
    base="$(basename "$trace")"
    [[ "$trace" == *"/msr_data/"* || "$trace" == *"/microsoft/"* || "$trace" == *"/msr/"* ]] && return 0
    [[ "$base" == hm_* || "$base" == proj_* || "$base" == usr_* || "$base" == src* || "$base" == prxy_* || "$base" == rsrch_* || "$base" == mds_* || "$base" == web_* || "$base" == stg_* || "$base" == ts_* || "$base" == wdev_* ]] && return 0
    return 1
}

# Returns: oracleGeneral | csv
get_trace_type() {
    local trace="$1"
    local base
    base="$(basename "$trace")"
    case "$base" in
        *.oracleGeneral.zst|*.oracleGeneral.bin|*.oracleGeneral)
            echo "oracleGeneral"
            ;;
        *)
            echo "csv"
            ;;
    esac
}

get_trace_params() {
    local trace="$1"
    if [ "$(get_trace_type "$trace")" = "oracleGeneral" ]; then
        echo ""
        return 0
    fi
    if is_msr_trace "$trace"; then
        echo "$TRACE_PARAMS_MSR"
    else
        echo "$TRACE_PARAMS_COMMON"
    fi
}

matches_sources() {
    local trace="$1"
    local sources="$2"
    local s
    [ -z "$sources" ] && return 0
    local lower
    lower="$(printf '%s' "$trace" | tr '[:upper:]' '[:lower:]')"
    for s in $sources; do
        s="$(printf '%s' "$s" | tr '[:upper:]' '[:lower:]')"
        [[ "$lower" == *"$s"* ]] && return 0
    done
    return 1
}

collect_traces() {
    local -n _out="$1"
    local sources="${2:-}"
    local f base
    _out=()
    # Follow nested directories and nested symlinks, and collect every
    # supported format. Previously CSV traces were ignored whenever even one
    # oracleGeneral trace existed.
    while IFS= read -r -d '' f; do
        base="$(basename "$f")"
        [[ "$base" == *".sample.csv"* ]] && continue
        matches_sources "$f" "$sources" || continue
        _out+=("$f")
    done < <(
        find -L "$DATA_DIR" -type f \
            \( -name '*.oracleGeneral.zst' \
               -o -name '*.oracleGeneral.bin' \
               -o -name '*.oracleGeneral' \
               -o -name '*.csv' \) \
            -print0 | sort -z
    )
}

# cachesim streams .zst when built with zstd; do NOT unpack 2TB to disk.
run_cachesim() {
    local trace="$1"
    local algo="$2"
    local cache_size="$3"
    local extra_e="$4"
    local out_file="$5"
    local ttype params
    ttype="$(get_trace_type "$trace")"
    params="$(get_trace_params "$trace")"
    rm -f "$(basename "$trace").cachesim" || true
    if [ -n "$params" ]; then
        "$CACHESIM" "$trace" "$ttype" "$algo" "$cache_size" \
            -t "$params" -e "$extra_e" > "$out_file" 2>&1 || true
    else
        "$CACHESIM" "$trace" "$ttype" "$algo" "$cache_size" \
            -e "$extra_e" > "$out_file" 2>&1 || true
    fi
    if [ -f "$(basename "$trace").cachesim" ]; then
        cat "$(basename "$trace").cachesim" >> "$out_file" || true
        rm -f "$(basename "$trace").cachesim" || true
    fi
}

# Extract miss ratios from cachesim stdout/.cachesim file.
# Typical line: "... miss ratio 0.12, byte miss ratio 0.34"
# Prints: "<omr> <bmr>"
extract_miss_pair() {
    local f="$1"
    local size_token="$2"   # e.g. 16  (from 16MB)
    [ -f "$f" ] || { echo " "; return 0; }

    local line=""
    line="$(grep -iE "cache size[[:space:]]+${size_token}[[:space:]]*(MiB|MB)?" "$f" 2>/dev/null | tail -n1 || true)"
    if [ -z "$line" ]; then
        line="$(grep -iE "miss ratio" "$f" 2>/dev/null | tail -n1 || true)"
    fi
    [ -n "$line" ] || { echo " "; return 0; }

    local omr bmr
    bmr="$(printf '%s\n' "$line" | sed -n 's/.*byte miss ratio[[:space:]]*\([0-9][0-9.]*\).*/\1/ip' | head -n1)"
    omr="$(printf '%s\n' "$line" | sed -n 's/.*miss ratio[[:space:]]*\([0-9][0-9.]*\).*byte miss ratio.*/\1/ip' | head -n1)"
    if [ -z "$omr" ]; then
        # No byte field on line — single "miss ratio" value.
        omr="$(printf '%s\n' "$line" | sed -n 's/.*miss ratio[[:space:]]*\([0-9][0-9.]*\).*/\1/ip' | head -n1)"
    fi
    if [ -n "$omr" ] && [ -z "$bmr" ]; then bmr="$omr"; fi
    if [ -n "$bmr" ] && [ -z "$omr" ]; then omr="$bmr"; fi
    echo "${omr} ${bmr}"
}

pick_metric() {
    local omr="$1" bmr="$2"
    if [ "$MISS_METRIC" = "omr" ]; then
        echo "$omr"
    else
        echo "$bmr"
    fi
}

# Prints the aligned shadow's overall "<omr> <bmr>" from the aggregate summary.
extract_shadow_pair() {
    local f="$1"
    [ -s "$f" ] || { echo " "; return 0; }
    awk '/TLCACHE_MAB_SUMMARY/ {
        for (i=1; i<=NF; i++) {
            split($i, a, "=")
            if (a[1] == "shadow_omr") omr=a[2]
            if (a[1] == "shadow_bmr") bmr=a[2]
        }
    } END { if (omr != "" && bmr != "") print omr, bmr }' "$f"
}

extract_mab_pair() {
    local f="$1"
    [ -s "$f" ] || { echo " "; return 0; }
    awk '/TLCACHE_MAB_SUMMARY/ {
        for (i=1; i<=NF; i++) {
            split($i, a, "=")
            if (a[1] == "mab_omr") omr=a[2]
            if (a[1] == "mab_bmr") bmr=a[2]
        }
    } END { if (omr != "" && bmr != "") print omr, bmr }' "$f"
}

# Append only useful lines from a run output into the human log.
append_summary() {
    local src="$1" dst="$2"
    [ -f "$src" ] || return 0
    grep -iE "TLCACHE_MAB_SUMMARY|error|fatal|abort|cannot|failed" "$src" 2>/dev/null \
        | tail -n 20 >> "$dst" || true
}

if [ ! -x "$CACHESIM" ] && [ ! -f "$CACHESIM" ]; then
    echo "ERROR: cachesim not found at $CACHESIM"
    exit 1
fi

TRAIN_LOG="$LOG_OUTPUT_DIR/training_phase.log"

if [ "$FRESH_START" = "1" ]; then
    echo "FRESH_START=1: removing generated matrices/configs/logs from prior training"
    for SIZE in $CACHE_SIZES; do
        rm -f "$POLICY_DIR/meta_policy_v3_${SIZE}.txt" \
              "$POLICY_DIR/profiler_config_v3_${SIZE}.txt" \
              "$POLICY_DIR/meta_policy_delta_v1_${SIZE}.txt" \
              "$POLICY_DIR/meta_policy_confirmed_v1_${SIZE}.txt"
    done
    rm -f "$TRAIN_LOG" \
          "$LOG_OUTPUT_DIR"/mab_train_*_learning.csv \
          "$LOG_OUTPUT_DIR"/mab_train_*_progress.csv \
          "$LOG_OUTPUT_DIR"/training_compare_*.csv \
          "$LOG_OUTPUT_DIR"/mab_diag_* \
          "$TRAIN_OUTPUT_DIR/regime_events.csv" \
          "$TRAIN_OUTPUT_DIR/regime_stats.csv" \
          "$TRAIN_OUTPUT_DIR/feature_samples.csv" \
          "$TRAIN_OUTPUT_DIR/feature_stats.csv" \
          "$TRAIN_OUTPUT_DIR/type_stats.csv" \
          "$TRAIN_OUTPUT_DIR/train_stats.csv"
fi

# Exact markers are written by this version. Legacy entries are reconstructed
# only when a trace emitted TLCACHE_MAB_SUMMARY and reached its final log line;
# a trace interrupted by power loss is therefore rerun.
declare -A COMPLETED_EXACT=()
declare -A COMPLETED_LEGACY=()
COMPLETED_EXACT_COUNT=0
COMPLETED_LEGACY_COUNT=0

load_completed_traces() {
    local log="$1"
    local marker cache trace key
    [ "$RESUME_FROM_LOG" = "1" ] || return 0
    [ "$FRESH_START" != "1" ] || return 0
    [ -s "$log" ] || return 0

    while IFS='|' read -r marker cache trace; do
        [ "$marker" = "TRACE_DONE" ] || continue
        [ -n "$cache" ] && [ -n "$trace" ] || continue
        key="$cache|$trace"
        if [ -z "${COMPLETED_EXACT[$key]+x}" ]; then
            COMPLETED_EXACT["$key"]=1
            COMPLETED_EXACT_COUNT=$((COMPLETED_EXACT_COUNT + 1))
        fi
    done < <(awk -F'|' '$1 == "TRACE_DONE" && NF >= 3 { print }' "$log")

    while IFS='|' read -r cache trace; do
        [ -n "$cache" ] && [ -n "$trace" ] || continue
        key="$cache|$trace"
        if [ -z "${COMPLETED_LEGACY[$key]+x}" ]; then
            COMPLETED_LEGACY["$key"]=1
            COMPLETED_LEGACY_COUNT=$((COMPLETED_LEGACY_COUNT + 1))
        fi
    done < <(
        awk '
            /^=== Cache: .* \| Training on / {
                line=$0
                sub(/^=== Cache: /, "", line)
                split(line, parts, " \\| Training on ")
                cache=parts[1]
                trace=parts[2]
                sub(/ \([^)]*\) ===$/, "", trace)
                complete=0
                next
            }
            /TLCACHE_MAB_SUMMARY/ {
                if (cache != "" && trace != "") complete=1
                next
            }
            /^  logs:/ {
                if (complete && cache != "" && trace != "")
                    print cache "|" trace
                cache=""
                trace=""
                complete=0
            }
        ' "$log"
    )
}

load_completed_traces "$TRAIN_LOG"

TRAIN_TRACES=()
collect_traces TRAIN_TRACES "$TRAIN_SOURCES"

if [ ${#TRAIN_TRACES[@]} -eq 0 ]; then
    echo "ERROR: no traces under $DATA_DIR"
    echo "Expected *.oracleGeneral.zst (preferred) or *.csv"
    echo "On the run machine: ln -s /mnt/d/data ./external_data"
    exit 1
fi

{
    echo "=========================================="
    if [ "$FRESH_START" = "1" ] || [ ! -s "$TRAIN_LOG" ]; then
        echo "Training Phase Started: $(date)"
    else
        echo "Training Phase Resumed: $(date)"
    fi
    echo "DATA_DIR=$DATA_DIR"
    echo "TRAIN_SOURCES=${TRAIN_SOURCES:-<all>}"
    echo "MAB_PARAMS=$MAB_PARAMS"
    echo "MISS_METRIC=$MISS_METRIC"
    echo "BASELINE=in-process ThreeLCache shadow"
    echo "Traces=${#TRAIN_TRACES[@]}"
    echo "First trace: ${TRAIN_TRACES[0]}"
    echo "Resume-from-log=$RESUME_FROM_LOG exact_done=$COMPLETED_EXACT_COUNT legacy_done=$COMPLETED_LEGACY_COUNT"
    echo "TRAIN LOGS (only): regime_stats.csv + feature_stats.csv + type_stats.csv"
    echo "=========================================="
} >> "$TRAIN_LOG"

echo "Training: ${#TRAIN_TRACES[@]} traces from $DATA_DIR | sizes: $CACHE_SIZES | metric: $MISS_METRIC"
echo "Paste logs: $TRAIN_OUTPUT_DIR/{regime,feature,type}_stats.csv"

for CACHE_SIZE in $CACHE_SIZES; do
    SIZE_TOKEN="${CACHE_SIZE%MB}"
    PRIMARY_FILE="$POLICY_DIR/meta_policy_v3_${CACHE_SIZE}.txt"
    CONFIG_FILE="$POLICY_DIR/profiler_config_v3_${CACHE_SIZE}.txt"
    TRAIN_LOG_PREFIX="$LOG_OUTPUT_DIR/mab_train_${CACHE_SIZE}"

    START_TRACE=""
    case "$CACHE_SIZE" in
        16MB) START_TRACE="$START_16MB" ;;
        32MB) START_TRACE="$START_32MB" ;;
        64MB) START_TRACE="$START_64MB" ;;
    esac

    HAS_COMPLETED_HISTORY=false
    if [ "$RESUME_FROM_LOG" = "1" ] && [ "$FRESH_START" != "1" ]; then
        for KEY in "${!COMPLETED_EXACT[@]}" "${!COMPLETED_LEGACY[@]}"; do
            if [[ "$KEY" == "$CACHE_SIZE|"* ]]; then
                HAS_COMPLETED_HISTORY=true
                break
            fi
        done
    fi

    # Never silently start a new matrix or a new cumulative stats history when
    # the progress log says this cache size was already partially trained.
    if [ "$HAS_COMPLETED_HISTORY" = true ]; then
        MISSING_RESUME_FILE=""
        for REQUIRED in "$PRIMARY_FILE" "$CONFIG_FILE" \
                        "$TRAIN_OUTPUT_DIR/regime_events.csv" \
                        "$TRAIN_OUTPUT_DIR/feature_samples.csv"; do
            if [ ! -s "$REQUIRED" ]; then
                MISSING_RESUME_FILE="$REQUIRED"
                break
            fi
        done
        if [ -n "$MISSING_RESUME_FILE" ]; then
            echo "ERROR: refusing to resume $CACHE_SIZE because this prior-run file is missing:"
            echo "  $MISSING_RESUME_FILE"
            echo "Restore the existing policy/config/raw stats, or use FRESH_START=1 intentionally."
            exit 1
        fi
    fi

    SHOULD_RUN=true
    if [ -n "$START_TRACE" ]; then
        SHOULD_RUN=false
    fi

    echo "==========================================" | tee -a "$TRAIN_LOG"
    echo "Starting Training Loop for Cache Size: $CACHE_SIZE" | tee -a "$TRAIN_LOG"
    echo "  policy-file=$PRIMARY_FILE" | tee -a "$TRAIN_LOG"
    if [ -n "$START_TRACE" ]; then
        echo "  Skipping until: $START_TRACE" | tee -a "$TRAIN_LOG"
    fi
    echo "==========================================" | tee -a "$TRAIN_LOG"

    for TRACE in "${TRAIN_TRACES[@]}"; do
        TRACE_NAME="$(basename "$TRACE")"

        if [ "$SHOULD_RUN" = false ]; then
            if [[ "$TRACE_NAME" == "$START_TRACE"* ]] || [[ "$TRACE" == *"$START_TRACE"* ]]; then
                SHOULD_RUN=true
            else
                echo "   -> [Skipped] $TRACE_NAME (Cache: $CACHE_SIZE)" >> "$TRAIN_LOG"
                continue
            fi
        fi

        if [ "$RESUME_FROM_LOG" = "1" ] && [ "$FRESH_START" != "1" ]; then
            EXACT_KEY="$CACHE_SIZE|$TRACE"
            LEGACY_KEY="$CACHE_SIZE|$TRACE_NAME"
            if [ -n "${COMPLETED_EXACT[$EXACT_KEY]+x}" ] ||
               [ -n "${COMPLETED_LEGACY[$LEGACY_KEY]+x}" ]; then
                echo "   -> [Already completed] $TRACE_NAME (Cache: $CACHE_SIZE)" >> "$TRAIN_LOG"
                continue
            fi
        fi

        LOCAL_TYPE=$(get_trace_type "$TRACE")
        echo "=== Cache: $CACHE_SIZE | Training on $TRACE_NAME ($LOCAL_TYPE) ===" | tee -a "$TRAIN_LOG"

        TMP_MAB="$(mktemp)"

        # Train / update matrix. Only regime/feature/type_stats are retained.
        # train-log-prefix still sets the cdt_logs/train/ output directory.
        run_cachesim "$TRACE" 3lcache-mab "$CACHE_SIZE" \
            "${MAB_PARAMS},profiler-mode=train,policy-file=${PRIMARY_FILE},config-file=${CONFIG_FILE},train-log-enable=0,train-log-prefix=${TRAIN_LOG_PREFIX},diag-enable=0" \
            "$TMP_MAB"

        append_summary "$TMP_MAB" "$TRAIN_LOG"

        MAB_PAIR=$(extract_mab_pair "$TMP_MAB")
        MAB_OMR=$(echo "$MAB_PAIR" | awk '{print $1}')
        MAB_BMR=$(echo "$MAB_PAIR" | awk '{print $2}')
        MAB_RES=$(pick_metric "$MAB_OMR" "$MAB_BMR")

        BASE_PAIR=$(extract_shadow_pair "$TMP_MAB")
        BASE_OMR=$(echo "$BASE_PAIR" | awk '{print $1}')
        BASE_BMR=$(echo "$BASE_PAIR" | awk '{print $2}')
        BASE_RES=$(pick_metric "$BASE_OMR" "$BASE_BMR")

        TRACE_SUCCESS=false
        if [ -n "$BASE_RES" ] && [ -n "$MAB_RES" ]; then
            DIFF=$(awk -v a="$MAB_RES" -v b="$BASE_RES" 'BEGIN { printf "%.6f", a - b }')
            echo "  miss($MISS_METRIC): shadow=$BASE_RES mab=$MAB_RES diff=$DIFF" | tee -a "$TRAIN_LOG"
            TRACE_SUCCESS=true
        else
            echo "  WARNING: could not parse MAB/shadow ratios for $TRACE_NAME @ $CACHE_SIZE" | tee -a "$TRAIN_LOG"
        fi

        rm -f "$TMP_MAB" "${TRACE_NAME}.cachesim" || true
        echo "  logs: $TRAIN_OUTPUT_DIR/{regime,feature,type}_stats.csv" | tee -a "$TRAIN_LOG"
        if [ "$TRACE_SUCCESS" = true ]; then
            echo "TRACE_DONE|$CACHE_SIZE|$TRACE" >> "$TRAIN_LOG"
        fi
    done

done

echo "Training completed." | tee -a "$TRAIN_LOG"

echo ""
echo "TRAIN logs (paste these):"
echo "  $TRAIN_OUTPUT_DIR/regime_stats.csv"
echo "  $TRAIN_OUTPUT_DIR/feature_stats.csv"
echo "  $TRAIN_OUTPUT_DIR/type_stats.csv"
echo "Policy table: $POLICY_DIR/meta_policy_v3_<SIZE>.txt"
echo "Progress log: $TRAIN_LOG"
