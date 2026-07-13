#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/_build/bin"

DATA_DIR="$SCRIPT_DIR/data"
TRACE_PARAMS="time-col=1,obj-id-is-num=true,obj-id-col=2,obj-size-col=3"
CACHE_SIZES="16MB 32MB 64MB 128MB 256MB 512MB"
MAB_PARAMS="arm_count=5,arm_strategy=frequency,mab_gamma=0.03"

# איסוף רשימת הטרייסים לפעולה
if [ "$#" -ge 1 ]; then
    TRACES=("$@")
else
    TRACES=()
    for f in "$DATA_DIR"/*.csv; do
        [ -e "$f" ] || continue
        if [[ "$(basename "$f")" == *".sample.csv"* ]]; then continue; fi
        TRACES+=("$(basename "$f")")
    done
fi

LOG_OUTPUT_DIR="$SCRIPT_DIR/cdt_logs"
mkdir -p "$LOG_OUTPUT_DIR"
CENTRAL_LOG="$LOG_OUTPUT_DIR/simulation_comparison.log"

# אתחול קובץ הלוג המרכזי
echo "==========================================================" > "$CENTRAL_LOG"
echo "📝 3LCache Original vs Victor EXP3 MAB Text Log 📝" >> "$CENTRAL_LOG"
echo "Execution Timestamp: $(date)" >> "$CENTRAL_LOG"
echo "==========================================================" >> "$CENTRAL_LOG"
echo "" >> "$CENTRAL_LOG"

echo "Running Experiments: 3LCache Baseline vs Victor MAB"
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
    echo "| Cache Size | 3LCache Baseline | Victor MAB | Diff (MAB - Base) | Winner |"
    echo "| --- | --- | --- | --- | --- |"

    TRACE_MAB_WINS=0
    TRACE_BASE_WINS=0

    for CACHE_SIZE in $CACHE_SIZES; do
        TMP_BASE=$(mktemp)
        TMP_MAB=$(mktemp)

        # שימוש בשמות המקוריים הרשומים בבנייה של הסימולטור שלכם
        ./cachesim "$TRACE" csv 3LCache "$CACHE_SIZE" -t "$TRACE_PARAMS" > "$TMP_BASE" 2>&1 || true
        ./cachesim "$TRACE" csv 3lcache-mab "$CACHE_SIZE" -t "$TRACE_PARAMS" -e "$MAB_PARAMS" > "$TMP_MAB" 2>&1 || true

        # חילוץ מתמטי של מדד הפספוסים האמיתי והסופי
        BASE_RES=$(awk '/miss ratio/ { for (i=1; i<=NF; i++) if ($(i) ~ /ratio/) { val=$(i+1); gsub(/[^0-9.]/, "", val); if (val!="") final_val=val } } END { print final_val }' "$TMP_BASE")
        MAB_RES=$(awk '/miss ratio/ { for (i=1; i<=NF; i++) if ($(i) ~ /ratio/) { val=$(i+1); gsub(/[^0-9.]/, "", val); if (val!="") final_val=val } } END { print final_val }' "$TMP_MAB")

        if [ -z "$BASE_RES" ] || [ -z "$MAB_RES" ] || [ "$BASE_RES" = "" ] || [ "$MAB_RES" = "" ]; then
            echo "| $CACHE_SIZE | ERROR | ERROR | - | - |"
            rm -f "$TMP_BASE" "$TMP_MAB"
            continue
        fi

        # חישוב הפערים וקביעת המנצח
        DIFF=$(awk -v a="$MAB_RES" -v b="$BASE_RES" 'BEGIN { printf "%.4f", a - b }')
        SIGN=$(awk -v d="$DIFF" 'BEGIN { if (d + 0 < -0.00001) print -1; else if (d + 0 > 0.00001) print 1; else print 0 }')

        if [ "$SIGN" = "-1" ]; then
            WINNER="MAB 🏆"
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

        # כתיבת מקטעי הזמן לתוך הלוג המרכזי
        echo "========================================================" >> "$CENTRAL_LOG"
        echo "📦 CACHE SIZE: $CACHE_SIZE | TRACE: $TRACE_NAME" >> "$CENTRAL_LOG"
        echo "📊 Final Miss Ratio -> Baseline: $BASE_RES | Victor MAB: $MAB_RES" >> "$CENTRAL_LOG"
        echo "========================================================" >> "$CENTRAL_LOG"
        
        echo "--- 🔴 ORIGINAL 3LCACHE TIMELINE ---" >> "$CENTRAL_LOG"
        grep "\[Baseline\]" "$TMP_BASE" >> "$CENTRAL_LOG" || true
        echo "" >> "$CENTRAL_LOG"

        echo "--- 🔵 VICTOR EXP3 MAB TIMELINE ---" >> "$CENTRAL_LOG"
        grep "\[Victor MAB\]" "$TMP_MAB" >> "$CENTRAL_LOG" || true
        echo "" >> "$CENTRAL_LOG"
        echo "--------------------------------------------------------" >> "$CENTRAL_LOG"
        echo "" >> "$CENTRAL_LOG"

        rm -f "$TMP_BASE" "$TMP_MAB"
    done

    echo "  -> MAB wins: $TRACE_MAB_WINS / 6, Baseline wins: $TRACE_BASE_WINS / 6"
    echo ""
done

TOTAL=$((TOTAL_MAB_WINS + TOTAL_BASE_WINS + TOTAL_TIES))
echo "============================================"
echo "OVERALL SUMMARY (across all traces and sizes)"
echo "  MAB wins:      $TOTAL_MAB_WINS / $TOTAL"
echo "  Baseline wins: $TOTAL_BASE_WINS / $TOTAL"
echo "  Ties:          $TOTAL_TIES / $TOTAL"
echo "============================================"
echo "✅ הניסוי הסתיים בהצלחה! קובץ הטקסט המרכזי מוכן בנתיב: cdt_logs/simulation_comparison.log"
