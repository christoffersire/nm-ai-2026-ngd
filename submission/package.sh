#!/bin/bash
# Package submission zip from trained model weights.
#
# Usage:
#   ./submission/package.sh <detector.pt> [classifier.pt] [output-name]
#
# Examples:
#   ./submission/package.sh weights/best.pt                              # detector-only
#   ./submission/package.sh weights/best.pt weights/cls-best.pt          # two-stage
#   ./submission/package.sh weights/best.pt weights/cls-best.pt sub-v2   # two-stage, custom name

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DETECTOR="${1:?Usage: $0 <detector.pt> [classifier.pt] [output-name]}"

# Detect if second arg is a .pt file (classifier) or output name
CLASSIFIER=""
OUTPUT_NAME="submission"

if [ -n "${2:-}" ]; then
    if [[ "$2" == *.pt ]]; then
        CLASSIFIER="$2"
        OUTPUT_NAME="${3:-submission}"
    else
        OUTPUT_NAME="$2"
    fi
fi

OUTPUT_ZIP="${PROJECT_DIR}/${OUTPUT_NAME}.zip"

# Staging directory
STAGING=$(mktemp -d)
trap "rm -rf $STAGING" EXIT

echo "=== Packaging submission ==="

# Copy run.py
cp "$SCRIPT_DIR/run.py" "$STAGING/run.py"

# Copy config
cp "$SCRIPT_DIR/config.json" "$STAGING/config.json"

# Copy detector weights
cp "$DETECTOR" "$STAGING/detector.pt"
WEIGHT_SIZE=$(du -m "$STAGING/detector.pt" | cut -f1)
echo "  Detector weights: ${WEIGHT_SIZE} MB"

# Copy classifier weights (if provided)
if [ -n "$CLASSIFIER" ]; then
    cp "$CLASSIFIER" "$STAGING/classifier.pt"
    CLS_SIZE=$(du -m "$STAGING/classifier.pt" | cut -f1)
    echo "  Classifier weights: ${CLS_SIZE} MB"
    echo "  Mode: two-stage"
else
    echo "  Mode: detector-only"
fi

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
