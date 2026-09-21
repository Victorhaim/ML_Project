#!/bin/bash
# Collect balanced four-arm regime outcomes, then train frozen LightGBM arm
# selector models. Each trace is promoted atomically so interrupted traces can
# be retried without duplicating rows in the central dataset.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/external_data}"
INTERMEDIATE_DIR="${INTERMEDIATE_DIR:-$SCRIPT_DIR/intermediate-results}"
POLICY_DIR="${POLICY_DIR:-$INTERMEDIATE_DIR}"
ARM_MODEL_DIR="${ARM_MODEL_DIR:-$POLICY_DIR/arm-selector}"
LOG_OUTPUT_DIR="${LOG_OUTPUT_DIR:-$SCRIPT_DIR/cdt_logs}"
CACHESIM="${CACHESIM:-$SCRIPT_DIR/_build/bin/cachesim}"
CACHE_SIZES="${CACHE_SIZES:-32MB}"
MAB_PARAMS="${MAB_PARAMS:-arm-count=4,arm-strategy=mode,mab-gamma=0.05,objective=byte-miss-ratio}"
TRAIN_SOURCES="${TRAIN_SOURCES:-}"
# 1 = wipe and start over. Unset = wipe only if leftover CSV is the old schema.
RESET_ARM_COLLECTION="${RESET_ARM_COLLECTION:-auto}"
TRAIN_ARM_MODELS="${TRAIN_ARM_MODELS:-1}"
# cachesim output is streamed through this filter and held in memory only.
# Nothing of it is written to disk, at any size.
KEEP_LINES="${KEEP_LINES:-TLCACHE_MAB_SUMMARY|TLCACHE_MAB_FIT_DONE|CACHESIM_EXIT=|[Ee][Rr][Rr][Oo][Rr]|terminate called|Assertion|Segmentation|std::bad|exception}"
KEEP_TAIL="${KEEP_TAIL:-40}"

if [ ! -x "$CACHESIM" ]; then
    echo "ERROR: cachesim not executable at $CACHESIM"
    exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: training data directory does not exist: $DATA_DIR"
    exit 1
fi

matches_sources() {
    local trace="$1" source lower
    [ -z "$TRAIN_SOURCES" ] && return 0
    lower="$(printf '%s' "$trace" | tr '[:upper:]' '[:lower:]')"
    for source in $TRAIN_SOURCES; do
        source="$(printf '%s' "$source" | tr '[:upper:]' '[:lower:]')"
        [[ "$lower" == *"$source"* ]] && return 0
    done
    return 1
}

get_trace_type() {
    case "$(basename "$1")" in
        *.oracleGeneral.zst|*.oracleGeneral.bin|*.oracleGeneral)
            echo "oracleGeneral"
            ;;
        *)
            echo "csv"
            ;;
    esac
}

collect_traces() {
    local -n output="$1"
    local trace base
    output=()
    shopt -s globstar nullglob
    for trace in "$DATA_DIR"/**/*.oracleGeneral.zst \
                 "$DATA_DIR"/**/*.oracleGeneral.bin \
                 "$DATA_DIR"/**/*.oracleGeneral; do
        [ -e "$trace" ] || continue
        [[ "$trace" == *"/cache_dataset/"* ]] && continue
        matches_sources "$trace" || continue
        output+=("$trace")
    done
    if [ ${#output[@]} -eq 0 ]; then
        for trace in "$DATA_DIR"/**/*.csv; do
            [ -e "$trace" ] || continue
            [[ "$trace" == *"/cache_dataset/"* ]] && continue
            base="$(basename "$trace")"
            [[ "$base" == *".sample.csv"* ]] && continue
            matches_sources "$trace" || continue
            output+=("$trace")
        done
    fi
}

# User order: msr_full, metastorage, cloudphysics, alibaba last.
# cache_dataset is skipped in collect_traces (no usable traces).
sort_traces_user_order() {
    local -n traces="$1"
    local -a msr=() meta=() cloud=() ali=() other=()
    local t rel
    for t in "${traces[@]}"; do
        rel="${t#"$DATA_DIR"/}"
        case "$rel" in
            msr_full/*|msr_full) msr+=("$t") ;;
            metastorage/*|metastorage) meta+=("$t") ;;
            cloudphysics/*|cloudphysics) cloud+=("$t") ;;
            alibaba/*|alibaba) ali+=("$t") ;;
            *) other+=("$t") ;;
        esac
    done
    _sort_bucket() {
        local -n _b="$1"
        if [ ${#_b[@]} -gt 0 ]; then
            IFS=$'\n'
            _b=($(printf '%s\n' "${_b[@]}" | LC_ALL=C sort))
            unset IFS
        fi
    }
    _sort_bucket msr
    _sort_bucket meta
    _sort_bucket cloud
    _sort_bucket other
    _sort_bucket ali
    traces=("${msr[@]+"${msr[@]}"}" "${meta[@]+"${meta[@]}"}" \
            "${cloud[@]+"${cloud[@]}"}" "${other[@]+"${other[@]}"}" \
            "${ali[@]+"${ali[@]}"}")
}

# New collect writes start_w0..end_w3. Old 4-probe CSVs do not.
dataset_is_current() {
    local f="$1" header
    [ -s "$f" ] || return 0
    header="$(head -n1 "$f")"
    [[ "$header" == *",start_w0,"* || "$header" == *",start_w0" ]] || return 1
    [[ "$header" == *",f_0,"* ]] || return 1
    [[ "$header" == *",delta_bmr,"* ]] || return 1
    return 0
}

wipe_collection() {
    local why="$1"
    echo "Starting fresh ($why)"
    rm -f "$DATASET" "$PROGRESS" "$POLICY_STATE" "$CONFIG_STATE"
    rm -f "$LOG_OUTPUT_DIR/train/regime_events.csv" \
          "$LOG_OUTPUT_DIR/train/regime_stats.csv" \
          "$LOG_OUTPUT_DIR/train/type_stats.csv" \
          "$LOG_OUTPUT_DIR/train/feature_samples.csv" \
          "$LOG_OUTPUT_DIR/train/feature_stats.csv"
}

append_atomic_dataset() {
    local source="$1" destination="$2"
    [ -s "$source" ] || return 0
    if [ ! -s "$destination" ]; then
        cp "$source" "$destination"
    else
        awk 'NR > 1' "$source" >> "$destination"
    fi
}

TRAIN_TRACES=()
collect_traces TRAIN_TRACES
if [ ${#TRAIN_TRACES[@]} -eq 0 ]; then
    echo "ERROR: no training traces found recursively under $DATA_DIR"
    exit 1
fi
sort_traces_user_order TRAIN_TRACES

mkdir -p "$ARM_MODEL_DIR" "$LOG_OUTPUT_DIR/train"
echo "Arm selector collection: ${#TRAIN_TRACES[@]} traces"
echo "Data root: $DATA_DIR"
echo "TRAIN stats: $LOG_OUTPUT_DIR/train/"
echo "Order: msr_full, metastorage, cloudphysics, alibaba last (skip cache_dataset)"

for CACHE_SIZE in $CACHE_SIZES; do
    MODEL_SUBDIR="$ARM_MODEL_DIR/$CACHE_SIZE"
    STATE_DIR="$MODEL_SUBDIR/collector-state"
    DATASET="$MODEL_SUBDIR/arm_training_samples.csv"
    PROGRESS="$MODEL_SUBDIR/collection_progress.log"
    POLICY_STATE="$STATE_DIR/meta_policy_v3_${CACHE_SIZE}.txt"
    CONFIG_STATE="$MODEL_SUBDIR/profiler_config_v3_${CACHE_SIZE}.txt"
    mkdir -p "$MODEL_SUBDIR" "$STATE_DIR"
    # Drop full-output logs left by older versions of this script.
    rm -f "$STATE_DIR"/.tmp_*.log "$STATE_DIR"/fit_*.log

    if [ "$RESET_ARM_COLLECTION" = "1" ]; then
        wipe_collection "RESET_ARM_COLLECTION=1"
    elif [ "$RESET_ARM_COLLECTION" = "auto" ] &&
         [ -s "$DATASET" ] && ! dataset_is_current "$DATASET"; then
        wipe_collection "old CSV schema"
    fi
    touch "$PROGRESS"

    if [ ! -s "$CONFIG_STATE" ] &&
       [ -s "$POLICY_DIR/profiler_config_v3_${CACHE_SIZE}.txt" ]; then
        cp "$POLICY_DIR/profiler_config_v3_${CACHE_SIZE}.txt" "$CONFIG_STATE"
    fi
    if [ ! -s "$POLICY_STATE" ] &&
       [ -s "$POLICY_DIR/meta_policy_v3_${CACHE_SIZE}.txt" ]; then
        cp "$POLICY_DIR/meta_policy_v3_${CACHE_SIZE}.txt" "$POLICY_STATE"
    fi

    echo "=== Collecting $CACHE_SIZE ==="
    for TRACE in "${TRAIN_TRACES[@]}"; do
        TRACE_REL="${TRACE#"$DATA_DIR"/}"
        MARKER="TRACE_DONE|$CACHE_SIZE|$TRACE_REL"
        if grep -Fqx "$MARKER" "$PROGRESS"; then
            echo "[Already completed] $TRACE_REL"
            continue
        fi

        TRACE_ID="$(printf '%s' "$TRACE_REL" | tr ', ' '__')"
        SAFE_NAME="$(printf '%s' "$TRACE_REL" | tr '/, ' '___')"
        TMP_ROOT="$STATE_DIR/.tmp_${SAFE_NAME}_$$"
        TMP_DATA="${TMP_ROOT}.csv"
        TMP_POLICY="${TMP_ROOT}.policy"
        TMP_CONFIG="${TMP_ROOT}.config"
        rm -f "$TMP_DATA" "$TMP_POLICY" "$TMP_CONFIG"
        [ ! -s "$POLICY_STATE" ] || cp "$POLICY_STATE" "$TMP_POLICY"
        [ ! -s "$CONFIG_STATE" ] || cp "$CONFIG_STATE" "$TMP_CONFIG"

        TRACE_TYPE="$(get_trace_type "$TRACE")"
        EXTRA_PARAMS="${MAB_PARAMS},profiler-mode=train,arm-selector=collect,arm-training-file=${TMP_DATA},arm-trace-id=${TRACE_ID},arm-cache-size-label=${CACHE_SIZE},policy-file=${TMP_POLICY},config-file=${TMP_CONFIG},diag-enable=0,train-log-enable=0,train-log-prefix=${LOG_OUTPUT_DIR}/mab_train_${CACHE_SIZE}"
        echo "[Running] $TRACE_REL"
        set +e
        if [ "$TRACE_TYPE" = "oracleGeneral" ]; then
            RUN_OUT="$(
                { "$CACHESIM" "$TRACE" "$TRACE_TYPE" 3lcache-mab "$CACHE_SIZE" \
                    -e "$EXTRA_PARAMS" 2>&1; echo "CACHESIM_EXIT=$?"; } \
                    | grep -E "$KEEP_LINES" | tail -n "$KEEP_TAIL"
            )"
        else
            RUN_OUT="$(
                { "$CACHESIM" "$TRACE" "$TRACE_TYPE" 3lcache-mab "$CACHE_SIZE" \
                    -t "time-col=1,obj-id-is-num=true,obj-id-col=2,obj-size-col=3" \
                    -e "$EXTRA_PARAMS" 2>&1; echo "CACHESIM_EXIT=$?"; } \
                    | grep -E "$KEEP_LINES" | tail -n "$KEEP_TAIL"
            )"
        fi
        set -e
        STATUS="$(printf '%s\n' "$RUN_OUT" | sed -n 's/^CACHESIM_EXIT=//p' | tail -n1)"
        STATUS="${STATUS:-1}"

        if [ "$STATUS" -ne 0 ] ||
           ! printf '%s\n' "$RUN_OUT" | grep -q "TLCACHE_MAB_SUMMARY"; then
            echo "ERROR: collection failed for $TRACE_REL (exit=$STATUS)"
            printf '%s\n' "$RUN_OUT" | sed 's/^/       | /'
            echo "       this trace was not marked complete"
            exit 1
        fi

        append_atomic_dataset "$TMP_DATA" "$DATASET"
        [ ! -s "$TMP_POLICY" ] || mv "$TMP_POLICY" "$POLICY_STATE"
        [ ! -s "$TMP_CONFIG" ] || mv "$TMP_CONFIG" "$CONFIG_STATE"
        echo "$MARKER" >> "$PROGRESS"
        rm -f "$TMP_DATA" "$TMP_POLICY" "$TMP_CONFIG"
    done

    if [ "$TRAIN_ARM_MODELS" = "1" ]; then
        if [ ! -s "$DATASET" ]; then
            echo "ERROR: no scored window rows collected for $CACHE_SIZE"
            exit 1
        fi
        MEAN_MODEL="$MODEL_SUBDIR/mean_model.txt"
        DOWNSIDE_MODEL="$MODEL_SUBDIR/downside_model.txt"
        FIT_TRACE="${TRAIN_TRACES[0]}"
        FIT_TYPE="$(get_trace_type "$FIT_TRACE")"
        echo "[Fitting] first LightGBM from $DATASET"
        set +e
        FIT_OUT="$(
            { "$CACHESIM" "$FIT_TRACE" "$FIT_TYPE" 3lcache-mab "$CACHE_SIZE" \
                -e "${MAB_PARAMS},arm-selector=fit,arm-training-file=${DATASET},arm-mean-model=${MEAN_MODEL},arm-downside-model=${DOWNSIDE_MODEL},arm-cache-size-label=${CACHE_SIZE},diag-enable=0,train-log-enable=0" \
                2>&1; echo "CACHESIM_EXIT=$?"; } \
                | grep -E "$KEEP_LINES" | tail -n "$KEEP_TAIL"
        )"
        set -e
        FIT_STATUS="$(printf '%s\n' "$FIT_OUT" | sed -n 's/^CACHESIM_EXIT=//p' | tail -n1)"
        FIT_STATUS="${FIT_STATUS:-1}"
        if [ "$FIT_STATUS" -ne 0 ] ||
           ! printf '%s\n' "$FIT_OUT" | grep -q "TLCACHE_MAB_FIT_DONE"; then
            echo "ERROR: arm-selector fit failed for $CACHE_SIZE"
            printf '%s\n' "$FIT_OUT" | sed 's/^/       | /'
            exit 1
        fi
        printf '%s\n' "$FIT_OUT" | grep "TLCACHE_MAB_FIT_DONE" || true
    fi
done

echo "Arm selector collection/training complete."
echo "TRAIN statistics (same matrices as before):"
echo "  $LOG_OUTPUT_DIR/train/regime_stats.csv"
echo "  $LOG_OUTPUT_DIR/train/type_stats.csv"
echo "  $LOG_OUTPUT_DIR/train/feature_stats.csv"
echo "  $LOG_OUTPUT_DIR/train/regime_events.csv"
echo "  $LOG_OUTPUT_DIR/train/feature_samples.csv"
