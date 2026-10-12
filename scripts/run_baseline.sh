#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then echo 'Usage: bash scripts/run_baseline.sh /path/TEST.zip [output_dir]'; exit 2; fi
repo="$(cd "$(dirname "$0")/.." && pwd)"
input="$1"
out="${2:-$repo/outputs/run}"
python_bin="${PYTHON_BIN:-python}"
config="${RING_CONFIG:-configs/default.yaml}"
mkdir -p "$out"
cd "$repo"
export PYTHONUNBUFFERED=1
exec > >(tee -a "$out/run.log") 2>&1
"$python_bin" -m ring_detector.cli audit --input "$input" --output "$out/audit"
"$python_bin" -m ring_detector.cli segment --manifest "$out/audit/manifest.csv" --config "$config" --output "$out/segments"
"$python_bin" -m ring_detector.cli predict --baseline-only --manifest "$out/audit/manifest.csv" --segments "$out/segments" --config "$config" --output "$out/proposals"
"$python_bin" -m ring_detector.cli review-export --proposals "$out/proposals" --output "$out/review"
"$python_bin" -m ring_detector.cli review-check --annotations "$out/review/annotations.csv" --cells "$out/proposals/cells.csv"
echo "BASELINE READY: $out/review/review.html"
