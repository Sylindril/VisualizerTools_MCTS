#!/usr/bin/env bash
# Batch-produce MCTS visualizations for every .jsonl under a top-level directory.
# Recursively finds all .jsonl files and creates viz/ subdirectories alongside them.
#
# Usage: ./batch_visualize.sh <top_level_dir>
#
# Example:
#   data/20260312_053005_lazy_regular/reddit/mcts_0.jsonl
#   → data/20260312_053005_lazy_regular/reddit/viz/mcts_0.html

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <top_level_dir>"
    exit 1
fi

TOP_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$TOP_DIR" ]; then
    echo "Error: $TOP_DIR is not a directory"
    exit 1
fi

# Collect all JSONL files
mapfile -t all_files < <(find "$TOP_DIR" -name "*.jsonl" -type f | sort)

if [ ${#all_files[@]} -eq 0 ]; then
    echo "No .jsonl files found under $TOP_DIR"
    exit 1
fi

echo "Found ${#all_files[@]} JSONL files under $TOP_DIR"
echo "---"

# Group by parent directory for progress reporting
declare -A dir_counts
for f in "${all_files[@]}"; do
    d="$(dirname "$f")"
    dir_counts["$d"]=$(( ${dir_counts["$d"]:-0} + 1 ))
done

echo "Directories:"
for d in $(echo "${!dir_counts[@]}" | tr ' ' '\n' | sort); do
    echo "  $d/ (${dir_counts[$d]} files)"
done
echo "---"

total=0
failed=0

for f in "${all_files[@]}"; do
    parent="$(dirname "$f")"
    viz_dir="$parent/viz"
    mkdir -p "$viz_dir"

    name="$(basename "$f" .jsonl)"
    out="$viz_dir/$name"

    echo "Processing $f ..."
    if python3 "$SCRIPT_DIR/visualizer_enhanced.py" "$f" --output_filename "$out" --debug 2>&1 | tail -1; then
        :
    else
        echo "  FAILED: $f"
        ((failed++)) || true
    fi
    ((total++)) || true
done

echo "---"
echo "Done. $total files processed, $failed failed."
echo "Visualizations in per-folder viz/ subdirectories."
