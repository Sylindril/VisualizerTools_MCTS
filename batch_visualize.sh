#!/usr/bin/env bash
# Batch-produce MCTS visualizations for every .jsonl in an input folder.
# Usage: ./batch_visualize.sh <input_folder> <output_folder>

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_folder> <output_folder>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
files=("$INPUT_DIR"/*.jsonl)
shopt -u nullglob

if [ ${#files[@]} -eq 0 ]; then
    echo "No .jsonl files found in $INPUT_DIR"
    exit 1
fi

echo "Found ${#files[@]} JSONL files in $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "---"

failed=0
for f in "${files[@]}"; do
    name="$(basename "$f" .jsonl)"
    out="$OUTPUT_DIR/$name"
    echo "Processing $name ..."
    if python3 "$SCRIPT_DIR/visualizer_enhanced.py" "$f" --output_filename "$out" --debug 2>&1 | tail -1; then
        :
    else
        echo "  FAILED: $name"
        ((failed++)) || true
    fi
done

echo "---"
echo "Done. ${#files[@]} files processed, $failed failed."
echo "HTML files in: $OUTPUT_DIR"
