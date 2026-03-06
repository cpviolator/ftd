#!/usr/bin/env bash
# ============================================================================
# Benchmark Output Parser — Scrapes raw logs into CSV files
# ============================================================================
#
# Parses output from both bench_pipeline and the test binaries into
# structured CSV files suitable for analysis.
#
# Usage:
#   bash parse_benchmarks.sh [LOG_DIR]
#
# Output files (in LOG_DIR):
#   summary.csv           — One row per benchmark run (total time, throughput, TFLOPS)
#   stages.csv            — One row per stage per run (stage name, min/mean/std)
#   pipelined.csv         — Pipelined vs sequential comparison
#
# Handles two output formats:
#   1. bench_pipeline: Per-stage timing tables + TFLOPS/throughput summary
#   2. test binaries: CUDA event-based total timing + per-batch breakdown
# ============================================================================

set -euo pipefail

LOG_DIR="${1:-.}"

if [[ ! -d "$LOG_DIR" ]]; then
    echo "Error: $LOG_DIR is not a directory"
    exit 1
fi

SUMMARY_CSV="$LOG_DIR/summary.csv"
STAGES_CSV="$LOG_DIR/stages.csv"
PIPELINED_CSV="$LOG_DIR/pipelined.csv"

echo "Parsing benchmark logs in: $LOG_DIR"
echo ""

# ============================================================================
# 1. Summary CSV — one row per benchmark run
# ============================================================================
echo "benchmark,suite,n_ant,n_beam,n_ch,n_time,K,batch,n_dm,ch_fuse,n_batch,total_mean_ms,throughput_hz,beams_per_ch_sec,tflops,speedup,streams" \
    > "$SUMMARY_CSV"

for logfile in "$LOG_DIR"/raw_*.log; do
    [[ -f "$logfile" ]] || continue
    basename=$(basename "$logfile" .log)
    # Extract label from filename: raw_LABEL_TIMESTAMP.log
    label=$(echo "$basename" | sed 's/^raw_//; s/_[0-9]\{8\}_[0-9]\{6\}$//')

    # Determine suite
    suite="unknown"
    case "$label" in
        visbf*) suite="visbf" ;;
        voltbf*) suite="voltbf" ;;
        dedisp*) suite="dedisp" ;;
    esac

    # Determine streams mode
    streams="single"
    case "$label" in
        *_ms*) streams="multi" ;;
        *_pipe*) streams="pipelined" ;;
    esac

    # ---- Parse bench_pipeline output ----
    # Extract dimensions from header: "n_ant=1664  n_beam=4096  n_ch=128  n_time=256"
    n_ant=$({ grep -oP 'n_ant=\K[0-9]+' "$logfile" || true; } | head -1)
    n_beam=$({ grep -oP 'n_beam=\K[0-9]+' "$logfile" || true; } | head -1)
    n_ch=$({ grep -oP 'n_ch=\K[0-9]+' "$logfile" || true; } | head -1)
    n_time=$({ grep -oP 'n_time=\K[0-9]+' "$logfile" || true; } | head -1)
    n_dm=$({ grep -oP 'n_dm=\K[0-9]+' "$logfile" || true; } | head -1)

    # For visibility: extract K and batch from HERK header
    K=$({ grep -oP 'K=\K[0-9]+' "$logfile" || true; } | head -1)
    batch=$({ grep -oP 'batch=\K[0-9]+' "$logfile" || true; } | head -1)

    # Fused VoltBF: extract ch_fuse and n_batch
    ch_fuse=$({ grep -oP 'ch_fuse=\K[0-9]+' "$logfile" || true; } | head -1)
    n_batch_val=$({ grep -oP 'n_batch=\K[0-9]+' "$logfile" || true; } | head -1)
    # Also try n_payloads_batch as fallback
    if [[ -z "$n_batch_val" ]]; then
        n_batch_val=$({ grep -oP 'n_payloads_batch=\K[0-9]+' "$logfile" || true; } | head -1)
    fi

    # Fallback: extract from test binary CLI output
    if [[ -z "$n_ant" ]]; then
        n_ant=$({ grep -oP 'NA=\K[0-9]+' "$logfile" || true; } | head -1)
    fi
    if [[ -z "$n_ch" ]]; then
        n_ch=$({ grep -oP 'Nf=\K[0-9]+' "$logfile" || true; } | head -1)
    fi
    if [[ -z "$n_time" ]]; then
        n_time=$({ grep -oP 'Nt=\K[0-9]+' "$logfile" || true; } | head -1)
    fi
    if [[ -z "$n_beam" ]]; then
        n_beam=$({ grep -oP 'Nb=\K[0-9]+' "$logfile" || true; } | head -1)
    fi

    # Extract Total line from timing table: "  Total    X.XX    X.XX    X.XX"
    total_mean=$({ grep -P '^\s+Total\s' "$logfile" || true; } | awk '{print $3}' | head -1)

    # Extract TFLOPS and throughput
    tflops=$({ grep -oP 'TFLOPS:\s*\K[0-9.]+' "$logfile" || true; } | head -1)
    throughput=$({ grep -oP 'Throughput:\s*\K[0-9.]+' "$logfile" || true; } | head -1)
    beams_per_ch_sec=$({ grep -oP 'Beams/ch/sec:\s*\K[0-9.e+]+' "$logfile" || true; } | head -1)

    # Extract speedup (used by dedisp and pipelined benchmarks)
    speedup=$({ grep -oP 'Speedup:\s*\K[0-9.]+' "$logfile" || true; } | head -1)

    # Fallback: multi-stream format "Multi-stream total    X.XX    X.XX    X.XX"
    # awk: $1=Multi-stream $2=total $3=min $4=mean $5=std
    if [[ -z "$total_mean" ]]; then
        total_mean=$({ grep -P 'Multi-stream total' "$logfile" || true; } | awk '{print $4}' | head -1)
    fi

    # Dedisp: use CUTLASS_FP8 mean as total (or CuBLAS if CUTLASS not present)
    if [[ -z "$total_mean" ]] && [[ "$suite" == "dedisp" ]]; then
        total_mean=$({ grep -P '^\s+CUTLASS_FP8\s' "$logfile" || true; } | awk '{print $3}' | head -1)
        if [[ -z "$total_mean" ]]; then
            total_mean=$({ grep -P '^\s+CuBLAS_FP32\s' "$logfile" || true; } | awk '{print $3}' | head -1)
        fi
    fi

    # Fallback: test binary timing "Total: X.XXX ms"
    # Use section headers to find the right Total line:
    #   visbf -> "Correlator Timing" or "Visibility" section
    #   voltbf -> "Beamformer Timing" or "Voltage" section
    if [[ -z "$total_mean" ]]; then
        section_pattern=""
        case "$suite" in
            visbf)  section_pattern="Correlator Timing|Visibility.*Timing" ;;
            voltbf) section_pattern="Beamformer Timing|Voltage.*Timing" ;;
        esac

        if [[ -n "$section_pattern" ]]; then
            # Find line number of section header, then grab the first Total: after it
            section_line=$({ grep -nP "$section_pattern" "$logfile" || true; } | head -1 | cut -d: -f1)
            if [[ -n "$section_line" ]]; then
                total_mean=$({ tail -n +"$section_line" "$logfile" | grep -P '^\s+Total:\s' | head -1 || true; } \
                    | { grep -oP '[0-9]+\.[0-9]+' || true; } | head -1)
            fi
        fi

        # Final fallback: just take the first Total: line
        if [[ -z "$total_mean" ]]; then
            total_mean=$({ grep -P '^\s+Total:\s' "$logfile" || true; } | { grep -oP '[0-9]+\.[0-9]+' || true; } | head -1)
        fi
    fi
    if [[ -z "$throughput" ]]; then
        throughput=$({ grep -oP 'Throughput:\s*\K[0-9.]+' "$logfile" || true; } | head -1)
    fi

    # Default empty fields
    n_ant="${n_ant:-}"
    n_beam="${n_beam:-}"
    n_ch="${n_ch:-}"
    n_time="${n_time:-}"
    K="${K:-}"
    batch="${batch:-}"
    n_dm="${n_dm:-}"
    ch_fuse="${ch_fuse:-}"
    n_batch_val="${n_batch_val:-}"
    total_mean="${total_mean:-}"
    throughput="${throughput:-}"
    beams_per_ch_sec="${beams_per_ch_sec:-}"
    tflops="${tflops:-}"
    speedup="${speedup:-}"

    echo "$label,$suite,$n_ant,$n_beam,$n_ch,$n_time,$K,$batch,$n_dm,$ch_fuse,$n_batch_val,$total_mean,$throughput,$beams_per_ch_sec,$tflops,$speedup,$streams" \
        >> "$SUMMARY_CSV"
done

echo "Written: $SUMMARY_CSV"

# ============================================================================
# 2. Stages CSV — one row per stage per run
# ============================================================================
echo "benchmark,suite,stage,min_ms,mean_ms,std_ms,pct_total" > "$STAGES_CSV"

for logfile in "$LOG_DIR"/raw_*.log; do
    [[ -f "$logfile" ]] || continue
    basename=$(basename "$logfile" .log)
    label=$(echo "$basename" | sed 's/^raw_//; s/_[0-9]\{8\}_[0-9]\{6\}$//')

    suite="unknown"
    case "$label" in
        visbf*) suite="visbf" ;;
        voltbf*) suite="voltbf" ;;
        dedisp*) suite="dedisp" ;;
    esac

    # Parse bench_pipeline timing table lines:
    # "  stage_name           X.XX       X.XX       X.XX     XX.X%"
    # Skip header lines (contain "Stage" or "---")
    # Skip "Total" line
    { grep -P '^\s+\S+\s+[0-9]+\.[0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.[0-9]+%' \
        "$logfile" 2>/dev/null || true; } | while read -r line; do
        stage=$(echo "$line" | awk '{print $1}')
        min_ms=$(echo "$line" | awk '{print $2}')
        mean_ms=$(echo "$line" | awk '{print $3}')
        std_ms=$(echo "$line" | awk '{print $4}')
        pct=$(echo "$line" | awk '{print $5}' | tr -d '%')

        echo "$label,$suite,$stage,$min_ms,$mean_ms,$std_ms,$pct" >> "$STAGES_CSV"
    done

    # Parse dedisp per-mode timing lines (no pct% column):
    # "  CuBLAS_FP32          XX.XX       XX.XX       XX.XX"
    # "  CUTLASS_FP8          XX.XX       XX.XX       XX.XX"
    if [[ "$suite" == "dedisp" ]]; then
        { grep -P '^\s+(CuBLAS_FP32|CUTLASS_FP8)\s' "$logfile" 2>/dev/null || true; } | while read -r line; do
            stage=$(echo "$line" | awk '{print $1}')
            min_ms=$(echo "$line" | awk '{print $2}')
            mean_ms=$(echo "$line" | awk '{print $3}')
            std_ms=$(echo "$line" | awk '{print $4}')
            echo "$label,$suite,$stage,$min_ms,$mean_ms,$std_ms," >> "$STAGES_CSV"
        done
    fi

    # Also parse test binary profile output (QUDA TimeProfile format):
    # "  H2D                 X.XXX s"
    { grep -P '^\s+(H2D|Compute|D2H|Preamble|Epilogue)\s' "$logfile" 2>/dev/null || true; } | while read -r line; do
        stage=$(echo "$line" | awk '{print $1}')
        time_s=$(echo "$line" | awk '{print $2}')
        # Convert seconds to ms if present
        if [[ -n "$time_s" ]]; then
            mean_ms=$(echo "$time_s" | awk '{printf "%.3f", $1 * 1000}')
            echo "$label,$suite,$stage,,${mean_ms},," >> "$STAGES_CSV"
        fi
    done
done

echo "Written: $STAGES_CSV"

# ============================================================================
# 3. Pipelined CSV — sequential vs pipelined comparison
# ============================================================================
echo "benchmark,mode,total_ms,per_payload_ms,throughput_hz,speedup" > "$PIPELINED_CSV"

for logfile in "$LOG_DIR"/raw_*pipe*.log "$LOG_DIR"/raw_*Pipe*.log; do
    [[ -f "$logfile" ]] || continue
    basename=$(basename "$logfile" .log)
    label=$(echo "$basename" | sed 's/^raw_//; s/_[0-9]\{8\}_[0-9]\{6\}$//')

    # bench_pipeline pipelined output format:
    #   "  Serial           XXX.X        X.XX      XX.XX Hz"
    #   "  Pipelined        XXX.X        X.XX      XX.XX Hz"
    #   "  Speedup: X.XXx"
    speedup=$(grep -oP 'Speedup:\s*\K[0-9.]+' "$logfile" | head -1 || true)

    { grep -P '^\s+(Serial|Pipelined)\s' "$logfile" 2>/dev/null || true; } | while read -r line; do
        mode=$(echo "$line" | awk '{print $1}')
        total=$(echo "$line" | awk '{print $2}')
        per_pl=$(echo "$line" | awk '{print $3}')
        tput=$(echo "$line" | awk '{print $4}')
        echo "$label,$mode,$total,$per_pl,$tput,${speedup:-}" >> "$PIPELINED_CSV"
    done

    # Test binary pipelined output format:
    #   "  Sequential:  X.XXX ms total, X.XXX ms/payload"
    #   "  Pipelined:   X.XXX ms total, X.XXX ms/payload"
    #   "  Speedup:     X.XXx"
    { grep -P '^\s+(Sequential|Pipelined):' "$logfile" 2>/dev/null || true; } | while read -r line; do
        mode=$(echo "$line" | awk -F: '{print $1}' | xargs)
        total=$(echo "$line" | grep -oP '[0-9.]+(?= ms total)' | head -1)
        per_pl=$(echo "$line" | grep -oP '[0-9.]+(?= ms/payload)' | head -1)
        if [[ -n "$total" && -n "$per_pl" ]]; then
            tput=$(echo "$total" | awk -v n="$per_pl" '{if (n > 0) printf "%.2f", 1000.0 / n; else print ""}')
            echo "$label,$mode,$total,$per_pl,$tput,${speedup:-}" >> "$PIPELINED_CSV"
        fi
    done
done

echo "Written: $PIPELINED_CSV"

# ============================================================================
# Print summary table to stdout
# ============================================================================
echo ""
echo "================================================================"
echo "  SUMMARY"
echo "================================================================"
echo ""

if command -v column &>/dev/null; then
    column -t -s',' "$SUMMARY_CSV"
else
    cat "$SUMMARY_CSV"
fi

echo ""

# Print stages if present
n_stage_rows=$(wc -l < "$STAGES_CSV")
if [[ "$n_stage_rows" -gt 1 ]]; then
    echo "================================================================"
    echo "  PER-STAGE BREAKDOWN"
    echo "================================================================"
    echo ""
    if command -v column &>/dev/null; then
        column -t -s',' "$STAGES_CSV"
    else
        cat "$STAGES_CSV"
    fi
    echo ""
fi

# Print pipelined if present
n_pipe_rows=$(wc -l < "$PIPELINED_CSV")
if [[ "$n_pipe_rows" -gt 1 ]]; then
    echo "================================================================"
    echo "  PIPELINED vs SEQUENTIAL"
    echo "================================================================"
    echo ""
    if command -v column &>/dev/null; then
        column -t -s',' "$PIPELINED_CSV"
    else
        cat "$PIPELINED_CSV"
    fi
    echo ""
fi

echo "CSV files in: $LOG_DIR/"
echo "  summary.csv     — Total time, throughput, TFLOPS per config"
echo "  stages.csv      — Per-stage min/mean/std/pct breakdown"
echo "  pipelined.csv   — Sequential vs pipelined comparison"
