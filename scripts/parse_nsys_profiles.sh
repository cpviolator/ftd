#!/usr/bin/env bash
# ============================================================================
# Nsys Profile Parser — Categorizes GPU kernel time by pipeline stage
# ============================================================================
#
# Reads per-kernel CSV files exported by run_nsys_profile.sh and groups
# kernels into functional categories (GEMM, FFT, Gridding, etc.).
#
# Usage:
#   bash parse_nsys_profiles.sh [LOG_DIR]
#
# Output:
#   nsys_summary.csv   — Per-benchmark, per-category totals
#
# Input files expected:
#   nsys_<label>_<timestamp>_kernels.csv   — from `nsys stats -r cuda_gpu_kern_sum`
#   nsys_<label>_<timestamp>_memcpy.csv    — from `nsys stats -r cuda_gpu_mem_time_sum`
# ============================================================================

set -euo pipefail

LOG_DIR="${1:-.}"

if [[ ! -d "$LOG_DIR" ]]; then
    echo "Error: $LOG_DIR is not a directory"
    exit 1
fi

SUMMARY_CSV="$LOG_DIR/nsys_summary.csv"

echo "Parsing nsys profile data in: $LOG_DIR"
echo ""

# ============================================================================
# Categorize a kernel name into a pipeline stage
# ============================================================================
categorize_kernel() {
    local name="$1"
    local name_lower
    name_lower=$(echo "$name" | tr '[:upper:]' '[:lower:]')

    # Triangle packing — check BEFORE Cast_Pack (pack_antisymmetrize_triangle
    # contains "pack" but is triangle output, and quda::HermToTri is triangle)
    if echo "$name_lower" | grep -qE 'triangle|antisymmetri|hermtotri|triangulatefromherm'; then
        echo "TrianglePack"
        return
    fi

    # GEMM / HERK (CUTLASS, cuBLAS GEMM, cuBLAS GEMV)
    if echo "$name_lower" | grep -qE 'cutlass|herk|gemm|sgemm|cgemm|gemv|xmma'; then
        echo "GEMM_HERK"
        return
    fi

    # FFT (cuFFT kernels)
    if echo "$name_lower" | grep -qE 'cufft|spvector|spradix|fft'; then
        echo "FFT"
        return
    fi

    # Imaging: gridding, phasors, precomputation, pillbox
    if echo "$name_lower" | grep -qE 'pillbox|grid.*scatter|gridding|phasors|precomp|imaging'; then
        echo "Imaging"
        return
    fi

    # Taper
    if echo "$name_lower" | grep -qE 'taper'; then
        echo "Taper"
        return
    fi

    # Beam extraction
    if echo "$name_lower" | grep -qE 'extract_beam|beam_extract'; then
        echo "BeamExtract"
        return
    fi

    # Cast / pack / interleave (data format conversion)
    if echo "$name_lower" | grep -qE 'cast|pack|deinterleave|interleave|convert|promote|fp8|fp16|fp32|qc_to_fp8|decode_qc'; then
        echo "Cast_Pack"
        return
    fi

    # Corner turn (transpose)
    if echo "$name_lower" | grep -qE 'corner_turn|transpose'; then
        echo "CornerTurn"
        return
    fi

    # Time integration / reduction
    if echo "$name_lower" | grep -qE 'time_integrate|reduce|pol_reduce|accumulate'; then
        echo "TimeIntegrate"
        return
    fi

    # memset / fill
    if echo "$name_lower" | grep -qE 'memset|fill_kernel'; then
        echo "Memset"
        return
    fi

    echo "Other"
}

# ============================================================================
# Parse a single kernel CSV file
#
# nsys stats cuda_gpu_kern_sum CSV format (varies by nsys version):
#   Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
#   or similar column ordering.
#
# We auto-detect columns by header line.
# ============================================================================
parse_kernel_csv() {
    local csv_file="$1"
    local label="$2"

    if [[ ! -s "$csv_file" ]]; then
        return
    fi

    # Find header line (first line containing "Name" and "Time")
    local header_line=""
    local header_num=0
    while IFS= read -r line; do
        header_num=$((header_num + 1))
        if echo "$line" | grep -qi 'Name' && echo "$line" | grep -qi 'Time'; then
            header_line="$line"
            break
        fi
    done < "$csv_file"

    if [[ -z "$header_line" ]]; then
        echo "  WARNING: No header found in $csv_file"
        return
    fi

    # Find column indices (0-based) for Total Time and Name
    local col_total_time=-1
    local col_instances=-1
    local col_name=-1
    local col_idx=0

    # Use comma as delimiter, handle quoted fields
    IFS=',' read -ra headers <<< "$header_line"
    for h in "${headers[@]}"; do
        local h_clean
        h_clean=$(echo "$h" | tr -d '"' | xargs)
        local h_lower
        h_lower=$(echo "$h_clean" | tr '[:upper:]' '[:lower:]')

        if [[ "$h_lower" == *"total time"* ]]; then
            col_total_time=$col_idx
        elif [[ "$h_lower" == "instances" || "$h_lower" == "count" ]]; then
            col_instances=$col_idx
        elif [[ "$h_lower" == "name" ]]; then
            col_name=$col_idx
        fi
        col_idx=$((col_idx + 1))
    done

    if [[ "$col_total_time" -lt 0 || "$col_name" -lt 0 ]]; then
        echo "  WARNING: Could not find 'Total Time' or 'Name' columns in $csv_file"
        echo "  Header: $header_line"
        return
    fi

    # Parse data lines — accumulate time per category
    # Use awk because kernel names can contain commas in template params
    # Strategy: Name is always the last column (nsys convention)
    tail -n +"$((header_num + 1))" "$csv_file" | while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        # Skip non-data lines (e.g. separator lines)
        echo "$line" | grep -q '^[[:space:]]*"*[0-9]' || continue

        # Extract fields: everything up to last comma-separated field is numeric columns,
        # last field (possibly quoted) is the kernel name.
        # nsys csv: last column is Name (may be quoted and contain commas for templates)
        local total_ns instances kern_name

        # Use awk to extract: col_total_time (0-based), col_instances, and everything from col_name onward
        total_ns=$(echo "$line" | awk -F',' -v c=$((col_total_time + 1)) '{gsub(/"/, "", $c); print $c}')
        if [[ "$col_instances" -ge 0 ]]; then
            instances=$(echo "$line" | awk -F',' -v c=$((col_instances + 1)) '{gsub(/"/, "", $c); print $c}')
        else
            instances="1"
        fi

        # Name: take from col_name to end of line (handles commas in template names)
        kern_name=$(echo "$line" | awk -F',' -v c=$((col_name + 1)) '{
            s = "";
            for (i = c; i <= NF; i++) {
                if (s != "") s = s ",";
                s = s $i;
            }
            gsub(/"/, "", s);
            print s;
        }')

        [[ -z "$kern_name" || -z "$total_ns" ]] && continue

        local category
        category=$(categorize_kernel "$kern_name")

        # Output: label, category, instances, total_ns, kernel_name
        echo "${label}|${category}|${instances}|${total_ns}|${kern_name}"
    done
}

# ============================================================================
# Parse a single memcpy CSV file
# ============================================================================
parse_memcpy_csv() {
    local csv_file="$1"
    local label="$2"

    if [[ ! -s "$csv_file" ]]; then
        return
    fi

    # Find header
    local header_line=""
    local header_num=0
    while IFS= read -r line; do
        header_num=$((header_num + 1))
        if echo "$line" | grep -qi 'Operation' || echo "$line" | grep -qi 'Total Time'; then
            header_line="$line"
            break
        fi
    done < "$csv_file"

    if [[ -z "$header_line" ]]; then
        return
    fi

    # Find column indices
    local col_total_time=-1
    local col_instances=-1
    local col_operation=-1
    local col_idx=0

    IFS=',' read -ra headers <<< "$header_line"
    for h in "${headers[@]}"; do
        local h_clean
        h_clean=$(echo "$h" | tr -d '"' | xargs)
        local h_lower
        h_lower=$(echo "$h_clean" | tr '[:upper:]' '[:lower:]')

        if [[ "$h_lower" == *"total time"* ]]; then
            col_total_time=$col_idx
        elif [[ "$h_lower" == "instances" || "$h_lower" == "count" ]]; then
            col_instances=$col_idx
        elif [[ "$h_lower" == "operation" || "$h_lower" == "name" || "$h_lower" == *"copy kind"* ]]; then
            col_operation=$col_idx
        fi
        col_idx=$((col_idx + 1))
    done

    if [[ "$col_total_time" -lt 0 ]]; then
        return
    fi

    tail -n +"$((header_num + 1))" "$csv_file" | while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        echo "$line" | grep -q '^[[:space:]]*"*[0-9]' || continue

        local total_ns instances op_name

        total_ns=$(echo "$line" | awk -F',' -v c=$((col_total_time + 1)) '{gsub(/"/, "", $c); print $c}')
        if [[ "$col_instances" -ge 0 ]]; then
            instances=$(echo "$line" | awk -F',' -v c=$((col_instances + 1)) '{gsub(/"/, "", $c); print $c}')
        else
            instances="1"
        fi
        if [[ "$col_operation" -ge 0 ]]; then
            op_name=$(echo "$line" | awk -F',' -v c=$((col_operation + 1)) '{gsub(/"/, "", $c); print $c}')
        else
            op_name="Memcpy"
        fi

        [[ -z "$total_ns" ]] && continue

        # Categorize memcpy type
        local category="Memcpy"
        local op_lower
        op_lower=$(echo "$op_name" | tr '[:upper:]' '[:lower:]')
        if echo "$op_lower" | grep -q 'htod\|host.*device\|h2d'; then
            category="Memcpy_H2D"
        elif echo "$op_lower" | grep -q 'dtoh\|device.*host\|d2h'; then
            category="Memcpy_D2H"
        elif echo "$op_lower" | grep -q 'dtod\|device.*device\|d2d'; then
            category="Memcpy_D2D"
        fi

        echo "${label}|${category}|${instances}|${total_ns}|${op_name}"
    done
}

# ============================================================================
# Main: process all nsys CSV files
# ============================================================================
TMPFILE=$(mktemp)
trap "rm -f $TMPFILE" EXIT

echo "Processing kernel and memcpy CSV files..."

for kern_csv in "$LOG_DIR"/nsys_*_kernels.csv; do
    [[ -f "$kern_csv" ]] || continue
    base=$(basename "$kern_csv" _kernels.csv)
    # Extract label: nsys_<label>_<timestamp> -> <label>
    label=$(echo "$base" | sed 's/^nsys_//; s/_[0-9]\{8\}_[0-9]\{6\}$//')

    echo "  Parsing kernels: $label"
    parse_kernel_csv "$kern_csv" "$label" >> "$TMPFILE"

    # Look for matching memcpy csv
    mem_csv="${kern_csv/_kernels.csv/_memcpy.csv}"
    if [[ -f "$mem_csv" ]]; then
        echo "  Parsing memcpy:  $label"
        parse_memcpy_csv "$mem_csv" "$label" >> "$TMPFILE"
    fi
done

if [[ ! -s "$TMPFILE" ]]; then
    echo ""
    echo "No nsys data found. Run run_nsys_profile.sh first."
    exit 0
fi

# ============================================================================
# Aggregate: group by (benchmark, category), sum time, compute percentages
# ============================================================================
echo ""
echo "benchmark,category,n_calls,total_time_us,avg_time_us,pct_total" > "$SUMMARY_CSV"

# First pass: compute total time per benchmark
awk -F'|' '
{
    label = $1
    category = $2
    instances = $3 + 0
    total_ns = $4 + 0

    key = label "|" category
    cat_time[key] += total_ns
    cat_calls[key] += instances
    bench_total[label] += total_ns
    benchmarks[label] = 1
    categories[key] = category
}
END {
    # Sort by benchmark then by descending time within each benchmark
    n = 0
    for (key in cat_time) {
        split(key, parts, "|")
        rows[n] = key
        row_label[n] = parts[1]
        row_time[n] = cat_time[key]
        n++
    }
    # Simple insertion sort by (label, -time)
    for (i = 1; i < n; i++) {
        j = i
        while (j > 0 && (row_label[j] < row_label[j-1] || \
               (row_label[j] == row_label[j-1] && row_time[j] > row_time[j-1]))) {
            tmp = rows[j]; rows[j] = rows[j-1]; rows[j-1] = tmp
            tmp = row_label[j]; row_label[j] = row_label[j-1]; row_label[j-1] = tmp
            tmp = row_time[j]; row_time[j] = row_time[j-1]; row_time[j-1] = tmp
            j--
        }
    }
    for (i = 0; i < n; i++) {
        key = rows[i]
        label = row_label[i]
        cat = categories[key]
        calls = cat_calls[key]
        total_us = cat_time[key] / 1000.0
        avg_us = (calls > 0) ? total_us / calls : 0
        total = bench_total[label]
        pct = (total > 0) ? (cat_time[key] / total) * 100.0 : 0

        printf "%s,%s,%d,%.1f,%.1f,%.1f\n", label, cat, calls, total_us, avg_us, pct
    }
}' "$TMPFILE" >> "$SUMMARY_CSV"

echo "Written: $SUMMARY_CSV"
echo ""

# ============================================================================
# Print summary table to stdout
# ============================================================================
echo "================================================================"
echo "  NSYS PROFILE SUMMARY"
echo "================================================================"
echo ""

if command -v column &>/dev/null; then
    column -t -s',' "$SUMMARY_CSV"
else
    cat "$SUMMARY_CSV"
fi

echo ""

# Per-benchmark totals
echo "--- Per-benchmark GPU total ---"
awk -F',' 'NR > 1 {
    t[$1] += $4
}
END {
    for (b in t) printf "  %-30s  %.1f us  (%.2f ms)\n", b, t[b], t[b] / 1000.0
}' "$SUMMARY_CSV"

echo ""
echo "CSV: $SUMMARY_CSV"

# ============================================================================
# Also produce per-kernel detail file (all kernels, unsummarized)
# ============================================================================
DETAIL_CSV="$LOG_DIR/nsys_detail.csv"
echo "benchmark,category,n_calls,total_time_us,kernel_name" > "$DETAIL_CSV"
awk -F'|' '{
    total_us = ($4 + 0) / 1000.0
    printf "%s,%s,%s,%.1f,%s\n", $1, $2, $3, total_us, $5
}' "$TMPFILE" >> "$DETAIL_CSV"

echo "Written: $DETAIL_CSV  (per-kernel detail)"
