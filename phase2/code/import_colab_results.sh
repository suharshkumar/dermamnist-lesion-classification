#!/bin/bash
# Usage: ./import_colab_results.sh /path/to/DDP_Phase2_results.zip
# Unpacks the Colab results (member dirs + v2_ensemble/) into run/artifacts/, copies JSON+PNG into DDP_Phase2/, rebuilds the report.
set -e
ZIP="$1"; ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ZIP" ] || { echo "zip not found: $ZIP"; exit 1; }
mkdir -p "$ROOT/run/artifacts"
unzip -o -q "$ZIP" -d "$ROOT/run/artifacts"
ls "$ROOT/run/artifacts"
mkdir -p "$ROOT/DDP_Phase2/results/colab" "$ROOT/DDP_Phase2/figures"
for d in "$ROOT"/run/artifacts/*/; do n=$(basename "$d"); [ -f "$d/results.json" ] && cp "$d/results.json" "$ROOT/DDP_Phase2/results/colab/${n}_results.json"; done
cp "$ROOT"/run/artifacts/v2_ensemble/*.json "$ROOT/DDP_Phase2/results/colab/" 2>/dev/null || true
cp "$ROOT"/run/artifacts/v2_ensemble/*.png "$ROOT/DDP_Phase2/figures/" 2>/dev/null || true
"$ROOT/run/.venv/bin/python" -I "$ROOT/DDP_Phase2/report_src/build_report.py"
