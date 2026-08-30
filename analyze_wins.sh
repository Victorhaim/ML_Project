#!/bin/bash
# Compare aligned WinLedger CSVs or print the accumulated result.
# Uses only POSIX awk; no Python dependency.
set -euo pipefail

usage() {
    echo "Usage:"
    echo "  $0 compare BASE.csv MAB.csv TRACE CACHE_SIZE METRIC MARGIN OUT_DIR RECORD_ALL"
    echo "  $0 confirm BASE.csv MAB.csv TRACE CACHE_SIZE METRIC MARGIN MIN_STREAK CANDIDATES.csv"
    echo "  $0 build-policy CANDIDATES.csv POLICY.txt SHRINK MAX_PER_TRACE MAX_ROWS"
    echo "  $0 report OUT_DIR"
}

MODE="${1:-}"

if [ "$MODE" = "compare" ]; then
    [ "$#" -eq 9 ] || { usage; exit 2; }
    BASE="$2"; MAB="$3"; TRACE="$4"; CACHE_SIZE="$5"
    METRIC="$6"; MARGIN="$7"; OUT_DIR="$8"; RECORD_ALL="$9"
    LEDGER="$OUT_DIR/win_ledger.csv"
    SUMMARY="$OUT_DIR/win_summary.csv"
    mkdir -p "$OUT_DIR"

    [ -s "$BASE" ] && [ -s "$MAB" ] || {
        echo "  skip: missing window data"
        exit 0
    }

    [ -f "$LEDGER" ] || echo "trace,cache_size,win_idx,seq_end,reqs,bytes_req,base_omr,mab_omr,base_bmr,mab_bmr,gain,gain_pct,verdict,dom_arm,matrix_rows,w0,w1,w2,w3,write_ratio,avg_request_size,size_variance,singleton_ratio,popularity_skewness,avg_frequency,object_diversity,avg_reuse_distance,sequentiality_ratio,out_cache_hit_rate,request_rate,burstiness_index,arrival_time_variance,working_set_byte_delta,scan_ratio" > "$LEDGER"
    [ -f "$SUMMARY" ] || echo "trace,cache_size,metric,margin,windows,wins,losses,ties,win_req_share,mean_gain_on_wins,max_gain,net_weighted_gain,overall_base,overall_mab" > "$SUMMARY"

    awk -F, -v OFS=, \
        -v trace="$TRACE" -v cache="$CACHE_SIZE" -v metric="$METRIC" \
        -v margin="$MARGIN" -v ledger="$LEDGER" -v summary="$SUMMARY" \
        -v record_all="$RECORD_ALL" '
        NR==FNR {
            if (FNR==1) next
            idx=$2
            base_omr[idx]=$8; base_bmr[idx]=$9
            base_reqs[idx]=$4; base_misses[idx]=$5
            base_bytes[idx]=$6; base_byte_misses[idx]=$7
            next
        }
        FNR==1 { next }
        {
            idx=$2
            if (!(idx in base_reqs) || $4 <= 0) next
            windows++; reqs=$4; total_reqs+=reqs
            bv=(metric=="omr" ? base_omr[idx] : base_bmr[idx])
            mv=(metric=="omr" ? $8 : $9)
            gain=bv-mv
            net+=gain*reqs
            b_req+=base_reqs[idx]; b_miss+=base_misses[idx]
            b_bytes+=base_bytes[idx]; b_byte_miss+=base_byte_misses[idx]
            m_req+=$4; m_miss+=$5; m_bytes+=$6; m_byte_miss+=$7

            if (gain >= margin) {
                verdict="win"; wins++; win_reqs+=reqs; gain_sum+=gain
                if (gain > max_gain) max_gain=gain
            } else if (gain <= -margin) {
                verdict="loss"; losses++
            } else {
                verdict="tie"; ties++
            }

            if (verdict=="win" || (record_all=="1" && verdict!="tie")) {
                pct=(bv>0 ? gain/bv*100.0 : 0)
                print trace,cache,idx,$3,$4,$6,base_omr[idx],$8,base_bmr[idx],$9, \
                      sprintf("%.6f",gain),sprintf("%.4f",pct),verdict,$12,$13, \
                      $14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27, \
                      $28,$29,$30,$31,$32 >> ledger
            }
        }
        END {
            share=(total_reqs ? win_reqs/total_reqs : 0)
            mean=(wins ? gain_sum/wins : 0)
            net_gain=(total_reqs ? net/total_reqs : 0)
            if (metric=="omr") {
                overall_base=(b_req ? b_miss/b_req : 0)
                overall_mab=(m_req ? m_miss/m_req : 0)
            } else {
                overall_base=(b_bytes ? b_byte_miss/b_bytes : 0)
                overall_mab=(m_bytes ? m_byte_miss/m_bytes : 0)
            }
            print trace,cache,metric,margin,windows+0,wins+0,losses+0,ties+0, \
                  sprintf("%.4f",share),sprintf("%.6f",mean), \
                  sprintf("%.6f",max_gain),sprintf("%.6f",net_gain), \
                  sprintf("%.6f",overall_base),sprintf("%.6f",overall_mab) >> summary
            printf "  windows=%d wins=%d losses=%d ties=%d | win_share=%.3f net=%.5f\n", \
                   windows,wins,losses,ties,share,net_gain
        }
    ' "$BASE" "$MAB"

elif [ "$MODE" = "confirm" ]; then
    [ "$#" -eq 9 ] || { usage; exit 2; }
    BASE="$2"; MAB="$3"; TRACE="$4"; CACHE_SIZE="$5"
    METRIC="$6"; MARGIN="$7"; MIN_STREAK="$8"; CANDIDATES="$9"
    mkdir -p "$(dirname "$CANDIDATES")"

    [ -s "$BASE" ] && [ -s "$MAB" ] || {
        echo "  confirm skip: missing window data"
        exit 0
    }

    [ -f "$CANDIDATES" ] || echo "trace,cache_size,start_win,end_win,support,mean_gain,mean_mab,dom_arm,w0,w1,w2,w3,write_ratio,avg_request_size,size_variance,singleton_ratio,popularity_skewness,avg_frequency,object_diversity,avg_reuse_distance,sequentiality_ratio,out_cache_hit_rate,request_rate,burstiness_index,arrival_time_variance,working_set_byte_delta,scan_ratio" > "$CANDIDATES"

    # A confirmed candidate is not a lucky window. It is one representative
    # averaged across MIN_STREAK or more consecutive windows that each beat
    # the aligned 3L baseline by MARGIN. Averaging features and weights across
    # the streak generalizes the regime instead of memorizing one timestamp.
    awk -F, -v OFS=, \
        -v trace="$TRACE" -v cache="$CACHE_SIZE" -v metric="$METRIC" \
        -v margin="$MARGIN" -v min_streak="$MIN_STREAK" \
        -v candidates="$CANDIDATES" '
        function reset_streak( i) {
            count=0; req_sum=0; gain_sum=0; mab_sum=0
            start_idx=0; end_idx=0; prev_idx=-2
            for (i=1; i<=19; i++) sums[i]=0
        }
        function emit_streak( i,dom,mx,v) {
            if (count < min_streak || req_sum <= 0) { reset_streak(); return }
            dom=0; mx=sums[1]/req_sum
            for (i=2; i<=4; i++) {
                v=sums[i]/req_sum
                if (v > mx) { mx=v; dom=i-1 }
            }
            printf "%s,%s,%d,%d,%d,%.9f,%.9f,%d", \
                   trace,cache,start_idx,end_idx,count,gain_sum/req_sum,mab_sum/req_sum,dom \
                   >> candidates
            for (i=1; i<=4; i++) printf ",%.9g",sums[i]/req_sum >> candidates
            for (i=5; i<=19; i++) printf ",%.9g",sums[i]/req_sum >> candidates
            printf "\n" >> candidates
            confirmed++
            reset_streak()
        }
        NR==FNR {
            if (FNR==1) next
            idx=$2
            base_omr[idx]=$8
            base_bmr[idx]=$9
            next
        }
        FNR==1 { reset_streak(); next }
        {
            idx=$2
            bv=(metric=="omr" ? base_omr[idx] : base_bmr[idx])
            mv=(metric=="omr" ? $8 : $9)
            gain=bv-mv
            reqs=$4+0
            is_win=((idx in base_omr) && reqs>0 && gain>=margin)

            if (!is_win || (count>0 && idx!=prev_idx+1)) emit_streak()
            if (is_win) {
                if (count==0) start_idx=idx
                end_idx=idx; prev_idx=idx; count++
                req_sum+=reqs; gain_sum+=gain*reqs; mab_sum+=mv*reqs
                # Raw MAB window: weights are columns 14..17; features 18..32.
                for (i=1; i<=4; i++) sums[i]+=$(13+i)*reqs
                for (i=5; i<=19; i++) sums[i]+=$(13+i)*reqs
            }
        }
        END {
            emit_streak()
            printf "  confirmed_regimes=%d (min_streak=%d margin=%.6f)\n", \
                   confirmed+0,min_streak,margin
        }
    ' "$BASE" "$MAB"

elif [ "$MODE" = "build-policy" ]; then
    [ "$#" -eq 6 ] || { usage; exit 2; }
    CANDIDATES="$2"; POLICY="$3"; SHRINK="$4"; MAX_PER_TRACE="$5"; MAX_ROWS="$6"
    mkdir -p "$(dirname "$POLICY")"

    [ -s "$CANDIDATES" ] || {
        rm -f "$POLICY"
        echo "  no confirmed candidates; policy not created"
        exit 0
    }

    TMP_SORTED="${POLICY}.candidates.tmp"
    # Strongest confirmed regimes first. Limit representatives per trace to
    # prevent a large trace (e.g. proj_1) from monopolizing the matrix.
    { IFS= read -r _header || true
      sort -t, -k6,6gr
    } < "$CANDIDATES" > "$TMP_SORTED"

    awk -F, -v OFS=, -v shrink="$SHRINK" \
        -v max_per_trace="$MAX_PER_TRACE" -v max_rows="$MAX_ROWS" '
        NF < 27 { next }
        kept >= max_rows { next }
        per_trace[$1] >= max_per_trace { next }
        {
            per_trace[$1]++
            kept++

            # Shrink learned weights toward uniform (mean weight 1.0). This
            # preserves the winning direction while reducing overfit to one
            # training streak. Normalize afterwards so weights sum to K=4.
            sumw=0
            for (i=1; i<=4; i++) {
                w[i]=(1.0-shrink)*$(8+i)+shrink
                sumw+=w[i]
            }
            scale=(sumw>0 ? 4.0/sumw : 1.0)

            # WorkloadProfiler matrix format:
            # 15 features, measured MAB miss rate, then four weights.
            for (i=13; i<=27; i++) {
                if (i>13) printf OFS
                printf "%.9g",$i
            }
            printf OFS "%.9g",$7
            for (i=1; i<=4; i++) printf OFS "%.9g",w[i]*scale
            printf "\n"
        }
        END {
            printf "  confirmed_policy_rows=%d\n",kept+0 > "/dev/stderr"
        }
    ' "$TMP_SORTED" > "${POLICY}.tmp"

    if [ -s "${POLICY}.tmp" ]; then
        mv "${POLICY}.tmp" "$POLICY"
    else
        rm -f "${POLICY}.tmp" "$POLICY"
    fi
    rm -f "$TMP_SORTED"

elif [ "$MODE" = "report" ]; then
    [ "$#" -eq 2 ] || { usage; exit 2; }
    OUT_DIR="$2"
    LEDGER="$OUT_DIR/win_ledger.csv"
    SUMMARY="$OUT_DIR/win_summary.csv"
    [ -s "$LEDGER" ] || { echo "No win ledger at $LEDGER"; exit 0; }

    awk -F, '
        NR==1 { next }
        $13=="win" {
            wins++; gain+=$11
            if ($11>max || wins==1) { max=$11; top=$0 }
            arm[$14]++
        }
        $13=="loss" { losses++ }
        END {
            print "============================================================"
            print "WIN LEDGER: " FILENAME
            print "============================================================"
            printf "Winning windows: %d\nLosing windows:  %d\n", wins, losses
            if (wins) {
                printf "Mean gain on wins: %.6f\nMaximum gain:      %.6f\n", gain/wins, max
                print "Dominant arms on wins:"
                for (a in arm) printf "  arm %s: %d windows\n", a, arm[a]
            }
        }
    ' "$LEDGER"

    if [ -s "$SUMMARY" ]; then
        echo ""
        echo "Per-trace summary:"
        awk -F, 'NR==1 {next} {
            printf "  %-28s %-6s windows=%-5s wins=%-5s losses=%-5s net=%s\n",
                   $1,$2,$5,$6,$7,$12
        }' "$SUMMARY"
    fi
else
    usage
    exit 2
fi
