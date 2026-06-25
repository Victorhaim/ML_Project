#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bash "$SCRIPT_DIR/scripts/install_libcachesim.sh"
cd "$SCRIPT_DIR/_build/bin"

DATA_DIR="$SCRIPT_DIR/data"
TRACE_PARAMS="time-col=1,obj-id-is-num=true,obj-id-col=2,obj-size-col=3"
CACHE_SIZES="1MB 10MB 50MB 100MB 200MB"
MAB_PARAMS="arm_count=8,arm_strategy=frequency,mab_gamma=0.1"

# Extract the miss-ratio value that follows the words "miss ratio" in cachesim output.
# Usage: get_miss <cachesim args...>
get_miss() {
    "$@" 2>&1 | awk '/miss ratio/ { for (i=1; i<=NF; i++) if ($i == "ratio") { print $(i+1); exit } }'
}

# Collect all CSV traces (skip .zip), or use args if provided.
if [ "$#" -ge 1 ]; then
    TRACES=("$@")
else
    TRACES=()
    for f in "$DATA_DIR"/*.csv; do
        TRACES+=("$(basename "$f")")
    done
fi

echo "Running Experiments: 3LCache Baseline vs 3LCache-MAB"
echo "MAB config: $MAB_PARAMS"
echo "Traces: ${#TRACES[@]}"
echo ""

TOTAL_MAB_WINS=0
TOTAL_BASE_WINS=0
TOTAL_TIES=0

for TRACE_NAME in "${TRACES[@]}"; do
    TRACE="$DATA_DIR/$TRACE_NAME"
    if [ ! -f "$TRACE" ]; then
        echo "Skipping $TRACE_NAME (not found)"
        continue
    fi

    echo "=== $TRACE_NAME ==="
    echo "| Cache Size | 3LCache Baseline | 3LCache MAB | Diff (MAB - Base) |"
    echo "| --- | --- | --- | --- |"

    TRACE_MAB_WINS=0
    TRACE_BASE_WINS=0

    for CACHE_SIZE in $CACHE_SIZES; do
        BASE_RES=$(get_miss ./cachesim "$TRACE" csv 3LCache "$CACHE_SIZE" -t "$TRACE_PARAMS")
        MAB_RES=$(get_miss  ./cachesim "$TRACE" csv 3lcache-mab "$CACHE_SIZE" -t "$TRACE_PARAMS" -e "$MAB_PARAMS")

        if [ -z "$BASE_RES" ] || [ -z "$MAB_RES" ]; then
            echo "| $CACHE_SIZE | ERROR | ERROR | - |"
            continue
        fi

        DIFF=$(awk -v a="$MAB_RES" -v b="$BASE_RES" 'BEGIN { printf "%.4f", a - b }')
        SIGN=$(awk -v d="$DIFF" 'BEGIN { if (d + 0 < 0) print -1; else if (d + 0 > 0) print 1; else print 0 }')

        if [ "$SIGN" = "-1" ]; then
            WINNER=MAB
            TRACE_MAB_WINS=$((TRACE_MAB_WINS + 1))
            TOTAL_MAB_WINS=$((TOTAL_MAB_WINS + 1))
        elif [ "$SIGN" = "1" ]; then
            WINNER=Base
            TRACE_BASE_WINS=$((TRACE_BASE_WINS + 1))
            TOTAL_BASE_WINS=$((TOTAL_BASE_WINS + 1))
        else
            WINNER=tie
            TOTAL_TIES=$((TOTAL_TIES + 1))
        fi

        echo "| $CACHE_SIZE | $BASE_RES | $MAB_RES | $DIFF ($WINNER) |"
    done

    echo "  -> MAB wins: $TRACE_MAB_WINS / 5, Baseline wins: $TRACE_BASE_WINS / 5"
    echo ""
done

TOTAL=$((TOTAL_MAB_WINS + TOTAL_BASE_WINS + TOTAL_TIES))
echo "============================================"
echo "OVERALL SUMMARY (across all traces and sizes)"
echo "  MAB wins:      $TOTAL_MAB_WINS / $TOTAL"
echo "  Baseline wins: $TOTAL_BASE_WINS / $TOTAL"
echo "  Ties:          $TOTAL_TIES / $TOTAL"
echo "============================================"
