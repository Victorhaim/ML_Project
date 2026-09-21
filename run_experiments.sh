#!/bin/bash
# Held-out TEST: one run contains both the MAB actor and an exactly aligned
# ThreeLCache shadow. Select legacy policy replay or a frozen arm booster.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# TEST is the held-out set: the ~41 traces committed under ./data, kept
# disjoint from the full external_data symlink that TRAIN uses. Prefer ./data;
# fall back to external_data only if ./data is absent. DATA_DIR overrides.
if [ -n "${DATA_DIR:-}" ]; then
    :
elif [ -d "$SCRIPT_DIR/data" ]; then
    DATA_DIR="$SCRIPT_DIR/data"
elif [ -d "$SCRIPT_DIR/external_data" ]; then
    DATA_DIR="$SCRIPT_DIR/external_data"
else
    DATA_DIR="$SCRIPT_DIR/data"
fi
INTERMEDIATE_DIR="$SCRIPT_DIR/intermediate-results"
# Read the trained policy + profiler config from the same place TRAIN wrote them.
# Override POLICY_DIR only to relocate both.
POLICY_DIR="${POLICY_DIR:-$INTERMEDIATE_DIR}"
LOG_OUTPUT_DIR="$SCRIPT_DIR/cdt_logs"
CACHESIM="${CACHESIM:-./_build/bin/cachesim}"

TEST_OUTPUT_DIR="$LOG_OUTPUT_DIR/test"
mkdir -p "$INTERMEDIATE_DIR" "$POLICY_DIR" "$LOG_OUTPUT_DIR" "$TEST_OUTPUT_DIR"
# TEST reports describe this invocation only. match_events.csv is the internal
# cumulative source used to rebuild the user-facing matrix after each trace.
rm -f "$TEST_OUTPUT_DIR/match_events.csv" \
      "$TEST_OUTPUT_DIR/match_stats.csv" \
      "$TEST_OUTPUT_DIR/type_stats.csv" \
      "$TEST_OUTPUT_DIR/arm_model_events.csv"

# CSV column mapping (ignored for oracleGeneral binary traces).
TRACE_PARAMS_COMMON="time-col=1,obj-id-is-num=true,obj-id-col=2,obj-size-col=3"
TRACE_PARAMS_MSR="time-col=1,obj-id-is-num=true,obj-id-col=5,obj-size-col=6"
# Optional vendor filters, space-separated path substrings.
# Example: TEST_SOURCES="twitter"
TEST_SOURCES="${TEST_SOURCES:-}"

# Default matches train_experiments.sh so TEST only evaluates sizes you trained.
# Override e.g. CACHE_SIZES="16MB 32MB 64MB" after those policies exist.
CACHE_SIZES="${CACHE_SIZES:-32MB}"
MAB_PARAMS="${MAB_PARAMS:-arm-count=4,arm-strategy=mode,mab-gamma=0.05,objective=byte-miss-ratio}"
MISS_METRIC="${MISS_METRIC:-bmr}"   # bmr | omr
# TEST refuses a nearest policy beyond this normalized distance and keeps the
# exact baseline path. Set <=0 to disable the ceiling for an ablation run.
TEST_MATCH_MAX_DIST="${TEST_MATCH_MAX_DIST:-3.5}"
# `model` uses a frozen first-stage LightGBM to choose one of the four queue
# sampling arms. `legacy` preserves policy-box replay for direct A/B runs.
ARM_SELECTOR="${ARM_SELECTOR:-model}" # model | legacy
ARM_MODEL_DIR="${ARM_MODEL_DIR:-$POLICY_DIR/arm-selector}"
ARM_RISK_LAMBDA="${ARM_RISK_LAMBDA:-0.5}"
if [ "$ARM_SELECTOR" != "legacy" ] && [ "$ARM_SELECTOR" != "model" ]; then
    echo "ERROR: ARM_SELECTOR must be legacy or model"
    exit 1
fi

# Held-out split is by data folder (./data vs external_data), so TEST_SOURCES
# is optional. Set it only to further narrow which ./data traces run.

is_msr_trace() {
    local trace="$1"
    local base
    base="$(basename "$trace")"
    [[ "$trace" == *"/msr_data/"* || "$trace" == *"/microsoft/"* || "$trace" == *"/msr/"* ]] && return 0
    [[ "$base" == hm_* || "$base" == proj_* || "$base" == usr_* || "$base" == src* || "$base" == prxy_* || "$base" == rsrch_* || "$base" == mds_* || "$base" == web_* || "$base" == stg_* || "$base" == ts_* || "$base" == wdev_* ]] && return 0
    return 1
}

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
    shopt -s globstar nullglob
    for f in "$DATA_DIR"/**/*.oracleGeneral.zst \
             "$DATA_DIR"/**/*.oracleGeneral.bin \
             "$DATA_DIR"/**/*.oracleGeneral; do
        [ -e "$f" ] || continue
        matches_sources "$f" "$sources" || continue
        _out+=("$f")
    done
    if [ ${#_out[@]} -eq 0 ]; then
        for f in "$DATA_DIR"/**/*.csv; do
            [ -e "$f" ] || continue
            base="$(basename "$f")"
            [[ "$base" == *".sample.csv"* ]] && continue
            matches_sources "$f" "$sources" || continue
            _out+=("$f")
        done
    fi
}

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

extract_miss_pair() {
    local f="$1"
    local size_token="$2"
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

if [ ! -x "$CACHESIM" ] && [ ! -f "$CACHESIM" ]; then
    echo "ERROR: cachesim not found at $CACHESIM"
    exit 1
fi

TEST_TRACES=()
collect_traces TEST_TRACES "$TEST_SOURCES"

CENTRAL_LOG="$LOG_OUTPUT_DIR/simulation_comparison.log"
TEST_CSV="$LOG_OUTPUT_DIR/test_compare_${MISS_METRIC}.csv"

cat << LOGHEADER > "$CENTRAL_LOG"
==========================================================
In-process ThreeLCache shadow vs TLCacheMAB (TEST)
Timestamp: $(date)
DATA_DIR=$DATA_DIR
TEST_SOURCES=${TEST_SOURCES:-<all>}
MAB_PARAMS=$MAB_PARAMS
TEST_MATCH_MAX_DIST=$TEST_MATCH_MAX_DIST
TEST_MATCH_WARMUP=full_feature_window
ARM_SELECTOR=$ARM_SELECTOR
ARM_RISK_LAMBDA=$ARM_RISK_LAMBDA
MISS_METRIC=$MISS_METRIC
Test traces: ${#TEST_TRACES[@]}
==========================================================

LOGHEADER

{
    echo "POLICY/CONFIG INPUTS:"
    for CACHE_SIZE in $CACHE_SIZES; do
        PRIMARY_FILE="$POLICY_DIR/meta_policy_v3_${CACHE_SIZE}.txt"
        CONFIG_FILE="$POLICY_DIR/profiler_config_v3_${CACHE_SIZE}.txt"
        if [ "$ARM_SELECTOR" = "legacy" ] && [ -s "$PRIMARY_FILE" ]; then
            echo "policy[$CACHE_SIZE]=$PRIMARY_FILE sha256=$(sha256sum "$PRIMARY_FILE" | awk '{print $1}')"
        elif [ "$ARM_SELECTOR" = "legacy" ]; then
            echo "policy[$CACHE_SIZE]=MISSING:$PRIMARY_FILE"
        else
            MODEL_SUBDIR="$ARM_MODEL_DIR/$CACHE_SIZE"
            MEAN_MODEL="$MODEL_SUBDIR/mean_model.txt"
            DOWNSIDE_MODEL="$MODEL_SUBDIR/downside_model.txt"
            MODEL_META="$MODEL_SUBDIR/metadata.txt"
            CONFIG_FILE="$MODEL_SUBDIR/profiler_config_v3_${CACHE_SIZE}.txt"
            for MODEL_FILE in "$MEAN_MODEL" "$DOWNSIDE_MODEL" "$MODEL_META"; do
                if [ -s "$MODEL_FILE" ]; then
                    echo "arm_model[$CACHE_SIZE]=$MODEL_FILE sha256=$(sha256sum "$MODEL_FILE" | awk '{print $1}')"
                else
                    echo "arm_model[$CACHE_SIZE]=MISSING:$MODEL_FILE"
                fi
            done
        fi
        if [ -s "$CONFIG_FILE" ]; then
            echo "config[$CACHE_SIZE]=$CONFIG_FILE sha256=$(sha256sum "$CONFIG_FILE" | awk '{print $1}')"
        else
            echo "config[$CACHE_SIZE]=MISSING:$CONFIG_FILE"
        fi
    done
    echo ""
} >> "$CENTRAL_LOG"

echo "trace,cache_size,base_${MISS_METRIC},mab_${MISS_METRIC},diff_mab_minus_base,winner,base_omr,base_bmr,mab_omr,mab_bmr" > "$TEST_CSV"
echo "Running TEST: in-process shadow vs MAB"
echo "Data root: $DATA_DIR"
echo "MAB config: $MAB_PARAMS"
echo "Arm selector: $ARM_SELECTOR"
if [ "$ARM_SELECTOR" = "legacy" ]; then
    echo "TEST match ceiling: $TEST_MATCH_MAX_DIST (farther candidates use baseline)"
    echo "TEST match warm-up: one full profiler feature window"
else
    echo "Arm model risk lambda: $ARM_RISK_LAMBDA"
    echo "Arm model warm-up: arm 3 until one full profiler feature window"
fi
echo "Metric: $MISS_METRIC (lower is better; diff = MAB - Base)"
echo "Test traces: ${#TEST_TRACES[@]}"
echo ""

TOTAL_MAB_WINS=0
TOTAL_BASE_WINS=0
TOTAL_TIES=0
TOTAL_ERRORS=0

TMP_MAB=""
cleanup() {
    [ -n "${TMP_MAB:-}" ] && rm -f "$TMP_MAB" 2>/dev/null || true
    rm -f ./*.cachesim 2>/dev/null || true
}
trap cleanup EXIT

if [ ${#TEST_TRACES[@]} -eq 0 ]; then
    echo "ERROR: no test traces under $DATA_DIR"
    echo "Set TEST_SOURCES or put *.oracleGeneral.zst under external_data/"
    exit 1
fi

for TRACE in "${TEST_TRACES[@]}"; do
    TRACE_NAME="$(basename "$TRACE")"
    TRACE_REL="${TRACE#"$DATA_DIR"/}"
    TRACE_ID="$(printf '%s' "$TRACE_REL" | tr ', ' '__')"
    if [ ! -f "$TRACE" ]; then
        echo "Skipping $TRACE_NAME (not found)"
        continue
    fi

    LOCAL_TYPE=$(get_trace_type "$TRACE")
    echo "=== $TRACE_NAME ($LOCAL_TYPE) ==="
    echo "| Cache Size | Base($MISS_METRIC) | MAB($MISS_METRIC) | Diff(MAB-Base) | Winner |"
    echo "| --- | --- | --- | --- | --- |"

    TRACE_MAB_WINS=0
    TRACE_BASE_WINS=0

    for CACHE_SIZE in $CACHE_SIZES; do
        SIZE_TOKEN="${CACHE_SIZE%MB}"
        PRIMARY_FILE="$POLICY_DIR/meta_policy_v3_${CACHE_SIZE}.txt"
        CONFIG_FILE="$POLICY_DIR/profiler_config_v3_${CACHE_SIZE}.txt"
        MODEL_SUBDIR="$ARM_MODEL_DIR/$CACHE_SIZE"
        MEAN_MODEL="$MODEL_SUBDIR/mean_model.txt"
        DOWNSIDE_MODEL="$MODEL_SUBDIR/downside_model.txt"
        MODEL_META="$MODEL_SUBDIR/metadata.txt"
        if [ "$ARM_SELECTOR" = "model" ]; then
            CONFIG_FILE="$MODEL_SUBDIR/profiler_config_v3_${CACHE_SIZE}.txt"
        fi

        if [ "$ARM_SELECTOR" = "legacy" ] && [ ! -s "$PRIMARY_FILE" ]; then
            echo "| $CACHE_SIZE | ERROR | ERROR | - | missing trained policy |"
            echo "ERROR: trained policy missing or empty: $PRIMARY_FILE" >> "$CENTRAL_LOG"
            TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
            continue
        fi
        # TEST distances are meaningful only with the exact trained feature
        # scale. Never silently fall back to constructor defaults.
        if [ ! -s "$CONFIG_FILE" ]; then
            echo "| $CACHE_SIZE | ERROR | ERROR | - | missing trained config |"
            echo "ERROR: trained profiler config missing or empty: $CONFIG_FILE" >> "$CENTRAL_LOG"
            TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
            continue
        fi
        if [ "$ARM_SELECTOR" = "model" ]; then
            if [ ! -s "$MEAN_MODEL" ] || [ ! -s "$DOWNSIDE_MODEL" ]; then
                echo "| $CACHE_SIZE | ERROR | ERROR | - | missing arm model |"
                echo "ERROR: arm selector models missing under $MODEL_SUBDIR" >> "$CENTRAL_LOG"
                TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
                continue
            fi
            if [ -s "$MODEL_META" ]; then
                META_SIZE="$(awk -F= '/^cache_size=/{print $2}' "$MODEL_META")"
                META_FEATS="$(awk -F= '/^feature_count=/{print $2}' "$MODEL_META")"
                if [ -n "$META_SIZE" ] && [ "$META_SIZE" != "$CACHE_SIZE" ]; then
                    echo "| $CACHE_SIZE | ERROR | ERROR | - | arm model cache_size mismatch |"
                    echo "ERROR: $MODEL_META cache_size=$META_SIZE" >> "$CENTRAL_LOG"
                    TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
                    continue
                fi
                if [ -n "$META_FEATS" ] && [ "$META_FEATS" != "25" ]; then
                    echo "| $CACHE_SIZE | ERROR | ERROR | - | arm model feature_count |"
                    echo "ERROR: $MODEL_META feature_count=$META_FEATS" >> "$CENTRAL_LOG"
                    TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
                    continue
                fi
            fi
        fi

        TMP_MAB="$(mktemp)"

        # One run: MAB actor + exact in-process ThreeLCache shadow.
        # oracleGeneral.zst is streamed by cachesim; no unpack.
        if [ "$ARM_SELECTOR" = "model" ]; then
            run_cachesim "$TRACE" 3lcache-mab "$CACHE_SIZE" \
                "${MAB_PARAMS},profiler-mode=test,arm-selector=model,arm-mean-model=${MEAN_MODEL},arm-downside-model=${DOWNSIDE_MODEL},arm-risk-lambda=${ARM_RISK_LAMBDA},arm-model-events-file=${TEST_OUTPUT_DIR}/arm_model_events.csv,arm-trace-id=${TRACE_ID},arm-cache-size-label=${CACHE_SIZE},policy-file=${PRIMARY_FILE},config-file=${CONFIG_FILE},diag-enable=0,train-log-enable=0,train-log-prefix=${LOG_OUTPUT_DIR}/mab_test_${CACHE_SIZE}" \
                "$TMP_MAB"
        else
            run_cachesim "$TRACE" 3lcache-mab "$CACHE_SIZE" \
                "${MAB_PARAMS},profiler-mode=test,arm-selector=legacy,test-match-max-dist=${TEST_MATCH_MAX_DIST},policy-file=${PRIMARY_FILE},config-file=${CONFIG_FILE},diag-enable=0,train-log-enable=0,train-log-prefix=${LOG_OUTPUT_DIR}/mab_test_${CACHE_SIZE}" \
                "$TMP_MAB"
        fi

        BASE_PAIR=$(extract_shadow_pair "$TMP_MAB")
        MAB_PAIR=$(extract_mab_pair "$TMP_MAB")
        BASE_OMR=$(echo "$BASE_PAIR" | awk '{print $1}')
        BASE_BMR=$(echo "$BASE_PAIR" | awk '{print $2}')
        MAB_OMR=$(echo "$MAB_PAIR" | awk '{print $1}')
        MAB_BMR=$(echo "$MAB_PAIR" | awk '{print $2}')
        BASE_RES=$(pick_metric "$BASE_OMR" "$BASE_BMR")
        MAB_RES=$(pick_metric "$MAB_OMR" "$MAB_BMR")

        if [ -z "$BASE_RES" ] || [ -z "$MAB_RES" ]; then
            echo "| $CACHE_SIZE | ERROR | ERROR | - | - |"
            {
                echo "PARSE ERROR: $TRACE_NAME @ $CACHE_SIZE"
                echo "--- MAB OUTPUT (tail) ---"
                tail -n 20 "$TMP_MAB" || true
                echo ""
            } >> "$CENTRAL_LOG"
            TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
            rm -f "$TMP_MAB"
            TMP_MAB=""
            continue
        fi

        DIFF=$(awk -v a="$MAB_RES" -v b="$BASE_RES" 'BEGIN { printf "%.6f", a - b }')
        SIGN=$(awk -v d="$DIFF" 'BEGIN { if (d+0 < -0.000001) print -1; else if (d+0 > 0.000001) print 1; else print 0 }')

        if [ "$SIGN" = "-1" ]; then
            WINNER="MAB"
            TRACE_MAB_WINS=$((TRACE_MAB_WINS + 1))
            TOTAL_MAB_WINS=$((TOTAL_MAB_WINS + 1))
        elif [ "$SIGN" = "1" ]; then
            WINNER="Base"
            TRACE_BASE_WINS=$((TRACE_BASE_WINS + 1))
            TOTAL_BASE_WINS=$((TOTAL_BASE_WINS + 1))
        else
            WINNER="tie"
            TOTAL_TIES=$((TOTAL_TIES + 1))
        fi

        echo "| $CACHE_SIZE | $BASE_RES | $MAB_RES | $DIFF | $WINNER |"
        echo "$TRACE_NAME,$CACHE_SIZE,$BASE_RES,$MAB_RES,$DIFF,$WINNER,$BASE_OMR,$BASE_BMR,$MAB_OMR,$MAB_BMR" >> "$TEST_CSV"
        {
            echo "========================================================"
            echo "CACHE: $CACHE_SIZE | TRACE: $TRACE_NAME"
            echo "metric=$MISS_METRIC base=$BASE_RES mab=$MAB_RES diff=$DIFF winner=$WINNER"
            echo "omr/bmr base=${BASE_OMR}/${BASE_BMR} mab=${MAB_OMR}/${MAB_BMR}"
            echo "========================================================"
            echo ""
        } >> "$CENTRAL_LOG"

        rm -f "$TMP_MAB"
        TMP_MAB=""
    done

    NSIZES=$(echo "$CACHE_SIZES" | wc -w | tr -d ' ')
    echo "  -> Summary: MAB wins $TRACE_MAB_WINS / $NSIZES, Baseline wins $TRACE_BASE_WINS / $NSIZES"
    echo ""
done

TOTAL=$((TOTAL_MAB_WINS + TOTAL_BASE_WINS + TOTAL_TIES))
echo "============================================"
echo "OVERALL SUMMARY"
echo "  MAB wins:      $TOTAL_MAB_WINS / $TOTAL"
echo "  Baseline wins: $TOTAL_BASE_WINS / $TOTAL"
echo "  Ties:          $TOTAL_TIES / $TOTAL"
echo "  Parse errors:  $TOTAL_ERRORS"
echo "============================================"

echo "Central log: $CENTRAL_LOG"
echo "Compare CSV: $TEST_CSV"
echo "TEST paste logs:"
echo "  $TEST_OUTPUT_DIR/match_events.csv  # per-window: type, size, delta, start arm"
echo "  $TEST_OUTPUT_DIR/match_stats.csv   # KNN distance bands (legacy); model=unmatched"
echo "  $TEST_OUTPUT_DIR/type_stats.csv    # type × size win/loss matrix"
echo "  $TEST_OUTPUT_DIR/arm_model_events.csv"
