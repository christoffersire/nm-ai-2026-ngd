#!/bin/bash
# Package submission zip from trained model weights.
#
# Usage:
#   ./submission/package.sh <path-to-best.pt> [output-name]
#
# Example:
#   ./submission/package.sh runs/detect/full-class-v1/weights/best.pt
#   ./submission/package.sh ~/Downloads/best.pt submission-v2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

WEIGHTS="${1:?Usage: $0 <path-to-best.pt> [output-name]}"
OUTPUT_NAME="${2:-submission}"
OUTPUT_ZIP="${PROJECT_DIR}/${OUTPUT_NAME}.zip"

# Staging directory
STAGING=$(mktemp -d)
trap "rm -rf $STAGING" EXIT

echo "=== Packaging submission ==="

# Copy run.py
cp "$SCRIPT_DIR/run.py" "$STAGING/run.py"

# Copy config
cp "$SCRIPT_DIR/config.json" "$STAGING/config.json"

# Copy model weights as detector.pt
cp "$WEIGHTS" "$STAGING/detector.pt"
WEIGHT_SIZE=$(du -m "$STAGING/detector.pt" | cut -f1)
echo "  Model weights: ${WEIGHT_SIZE} MB"

# Build zip
cd "$STAGING"
rm -f "$OUTPUT_ZIP"
zip -r "$OUTPUT_ZIP" . -x ".*" "__MACOSX/*"

echo ""
echo "=== Validating ==="
python3 "$SCRIPT_DIR/validate_submission.py" "$OUTPUT_ZIP"

echo ""
FINAL_SIZE=$(du -m "$OUTPUT_ZIP" | cut -f1)
echo "=== Done ==="
echo "  Output: $OUTPUT_ZIP (${FINAL_SIZE} MB compressed)"
