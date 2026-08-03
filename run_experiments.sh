#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

DATA_DIR="$SCRIPT_DIR/data"
INTERMEDIATE_DIR="$SCRIPT_DIR/intermediate-results"

# 1. תיקון מיפוי העמודות (2 ו-3) ושימוש במקפים (-) עבור MAB
TRACE_PARAMS="time-col=1,obj-id-is-num=true,obj-id-col=2,obj-size-col=3"
CACHE_SIZES="16MB 32MB 64MB"
MAB_PARAMS="arm-count=5,arm-strategy=frequency,mab-gamma=0.03"

# איסוף רשימת הטרייסים לטסט (ללא msr_data וללא sample)
TEST_TRACES=()
shopt -s globstar nullglob
for f in "$DATA_DIR"/**/*.csv; do
    [ -e "$f" ] || continue
    case "$f" in
        *"/msr_data/"* ) continue ;;
    esac
    base="$(basename "$f")"
    [[ "$base" == *".sample.csv"* ]] && continue
    TEST_TRACES+=("$f")
done

LOG_OUTPUT_DIR="$SCRIPT_DIR/cdt_logs"
mkdir -p "$LOG_OUTPUT_DIR"
CENTRAL_LOG="$LOG_OUTPUT_DIR/simulation_comparison.log"

# אתחול קובץ הלוג המרכזי
cat << EOF > "$CENTRAL_LOG"
==========================================================
📝 3LCache Baseline vs Victor EXP3 MAB (Test Mode) Log 📝
Execution Timestamp: $(date)
==========================================================

EOF

echo "🚀 Running Test Experiments: 3LCache Baseline vs Victor MAB (Trained Policy)"
echo "⚙️ MAB config: $MAB_PARAMS"
echo "📁 Traces: ${#TEST_TRACES[@]}"
echo ""

TOTAL_MAB_WINS=0
TOTAL_BASE_WINS=0
TOTAL_TIES=0

# מנגנון ניקוי בטוח לקבצים זמניים
TMP_BASE=""
TMP_MAB=""
cleanup() {
    [ -n "$TMP_BASE" ] && rm -f "$TMP_BASE" "${TMP_BASE}.cachesim" "${TMP_BASE}.parse" 2>/dev/null || true
    [ -n "$TMP_MAB" ] && rm -f "$TMP_MAB" "${TMP_MAB}.cachesim" 2>/dev/null || true
}
trap cleanup EXIT

for TRACE in "${TEST_TRACES[@]}"; do
    TRACE_NAME="$(basename "$TRACE")"
    if [ ! -f "$TRACE" ]; then
        echo "⚠️ Skipping $TRACE_NAME (not found)"
        continue
    fi

    echo "=== 📂 $TRACE_NAME ==="
    echo "| Cache Size | 3LCache Baseline | Victor MAB (Test) | Diff (MAB - Base) | Winner |"
    echo "| --- | --- | --- | --- | --- |"

    TRACE_MAB_WINS=0
    TRACE_BASE_WINS=0

    for CACHE_SIZE in $CACHE_SIZES; do
        POLICY_FILE="$INTERMEDIATE_DIR/meta-policy-${CACHE_SIZE}.txt"
        CONFIG_FILE="$INTERMEDIATE_DIR/profiler-config-${CACHE_SIZE}.txt"

        if [ ! -f "$POLICY_FILE" ]; then
            echo "| $CACHE_SIZE | MISSING POLICY | MISSING POLICY | - | SKIP |"
            continue
        fi

        TMP_BASE=$(mktemp)
        TMP_MAB=$(mktemp)

        # 1. הרצת Baseline (3lcache מקורי)
        rm -f "$(basename "$TRACE").cachesim" || true
        "./_build/bin/cachesim" "$TRACE" csv 3lcache "$CACHE_SIZE" -t "$TRACE_PARAMS" > "$TMP_BASE" 2>&1 || true
        
        OUTFILE_BASE="$(basename "$TRACE").cachesim"
        if [ -f "$OUTFILE_BASE" ]; then
            cp "$OUTFILE_BASE" "${TMP_BASE}.cachesim" || true
            echo "[INFO] saved baseline ofile: ${TMP_BASE}.cachesim" >> "$CENTRAL_LOG"
        fi

        # 2. הרצת Victor MAB במצב TEST עם מקפים תקינים בפרמטרים
        rm -f "$(basename "$TRACE").cachesim" || true
        "./_build/bin/cachesim" "$TRACE" csv 3lcache-mab "$CACHE_SIZE" \
            -t "$TRACE_PARAMS" \
            -e "${MAB_PARAMS},profiler-mode=test,policy-file=${POLICY_FILE},config-file=${CONFIG_FILE}" \
            > "$TMP_MAB" 2>&1 || true
            
        OUTFILE_MAB="$(basename "$TRACE").cachesim"
        if [ -f "$OUTFILE_MAB" ]; then
            cp "$OUTFILE_MAB" "${TMP_MAB}.cachesim" || true
            echo "[INFO] saved MAB ofile: ${TMP_MAB}.cachesim" >> "$CENTRAL_LOG"
        fi

        # פונקציית חילוץ מבוססת Regex חסינה
        extract_miss_by_kind() {
            local f="$1"; local size_mb="$2"; local kind="$3"
            if [ "$kind" = "mab" ]; then
                grep -iE "MAB.*cache size[[:space:]]+${size_mb}MiB" "$f" 2>/dev/null | tail -n1 | sed -n 's/.*miss ratio[^0-9]*\([0-9.]*\).*/\1/p' || true
            else
                grep -iE "cache size[[:space:]]+${size_mb}MiB" "$f" 2>/dev/null | grep -iv "MAB" | tail -n1 | sed -n 's/.*miss ratio[^0-9]*\([0-9.]*\).*/\1/p' || true
            fi
        }

        # חילוץ תוצאות
        BASE_RES=$(extract_miss_by_kind "${TMP_BASE}.cachesim" "${CACHE_SIZE%MB}" "base")
        [ -z "$BASE_RES" ] && BASE_RES=$(extract_miss_by_kind "$TMP_BASE" "${CACHE_SIZE%MB}" "base")

        MAB_RES=$(extract_miss_by_kind "${TMP_MAB}.cachesim" "${CACHE_SIZE%MB}" "mab")
        [ -z "$MAB_RES" ] && MAB_RES=$(extract_miss_by_kind "$TMP_MAB" "${CACHE_SIZE%MB}" "mab")

        # בדיקת אפסים תקינה מתמטית (אם שניהם 0.0000)
        IS_ZERO=$(awk -v a="$BASE_RES" -v b="$MAB_RES" 'BEGIN { if (a+0 == 0 && b+0 == 0) print 1; else print 0 }')
        if [ "$IS_ZERO" = "1" ]; then
            echo "[WARN] both miss ratios are 0. Running parse diagnostic for $TRACE_NAME $CACHE_SIZE" >> "$CENTRAL_LOG"
            "./_build/bin/cachesim" "$TRACE" csv 3lcache "$CACHE_SIZE" -t "$TRACE_PARAMS" -e "print" > "${TMP_BASE}.parse" 2>&1 || true
            head -n 40 "${TMP_BASE}.parse" >> "$CENTRAL_LOG" || true
        fi

        if [ -z "$BASE_RES" ] || [ -z "$MAB_RES" ]; then
            echo "| $CACHE_SIZE | ERROR | ERROR | - | - |"
            cleanup
            continue
        fi

        # חישוב הפערים וקביעת המנצח
        DIFF=$(awk -v a="$MAB_RES" -v b="$BASE_RES" 'BEGIN { printf "%.4f", a - b }')
        SIGN=$(awk -v d="$DIFF" 'BEGIN { if (d + 0 < -0.0001) print -1; else if (d + 0 > 0.0001) print 1; else print 0 }')

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
        {
            echo "========================================================"
            echo "📦 CACHE SIZE: $CACHE_SIZE | TRACE: $TRACE_NAME"
            echo "📊 Final Miss Ratio -> Baseline: $BASE_RES | Victor MAB: $MAB_RES"
            echo "========================================================"
            echo "--- 🔴 ORIGINAL 3LCACHE TIMELINE ---"
            grep "\[Baseline\]" "$TMP_BASE" || true
            echo ""
            echo "--- 🔵 VICTOR EXP3 MAB TIMELINE ---"
            grep -E "\[Victor MAB\]|\[Test\]" "$TMP_MAB" || true
            echo ""
            echo "--------------------------------------------------------"
            echo ""
        } >> "$CENTRAL_LOG"

        cleanup
    done

    echo "  -> Summary: MAB wins $TRACE_MAB_WINS / 3, Baseline wins $TRACE_BASE_WINS / 3"
    echo ""
done

TOTAL=$((TOTAL_MAB_WINS + TOTAL_BASE_WINS + TOTAL_TIES))
echo "============================================"
echo "📊 OVERALL SUMMARY (across all test traces and sizes)"
echo "   MAB wins:      $TOTAL_MAB_WINS / $TOTAL"
echo "   Baseline wins: $TOTAL_BASE_WINS / $TOTAL"
echo "   Ties:          $TOTAL_TIES / $TOTAL"
echo "============================================"
echo "✅ התהליך הסתיים בהצלחה! הלוג המרכזי מוכן בנתיב: $CENTRAL_LOG"