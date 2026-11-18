#!/bin/bash
# Usage: phase3/import_phase3_results.sh /path/to/DDP_Phase3_results_224.zip
# Unpacks the Colab Phase-3 archive (run dirs + ensemble/) into run/artifacts_phase3/ and copies JSON + PNG into DDP_Phase2/phase3/results|figures.
set -e
ZIP="$1"; ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$ZIP" ] || { echo "zip not found: $ZIP"; exit 1; }
DEST="$ROOT/run/artifacts_phase3"; mkdir -p "$DEST"
unzip -o -q "$ZIP" -d "$DEST"; ls "$DEST"
mkdir -p "$ROOT/DDP_Phase2/phase3/results" "$ROOT/DDP_Phase2/phase3/figures"
for d in "$DEST"/*/; do n=$(basename "$d"); [ -f "$d/results.json" ] && cp "$d/results.json" "$ROOT/DDP_Phase2/phase3/results/${n}_results.json"; done
cp "$DEST"/ensemble/*.json "$ROOT/DDP_Phase2/phase3/results/" 2>/dev/null || true
cp "$DEST"/ensemble/*.png "$ROOT/DDP_Phase2/phase3/figures/" 2>/dev/null || true
echo "imported into $DEST"
