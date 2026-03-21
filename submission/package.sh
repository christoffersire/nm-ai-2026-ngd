#!/bin/bash
# Package submission zip from trained model weights.
#
# Usage:
#   ./submission/package.sh <detector> [classifier] [output-name]
#
# Examples:
#   ./submission/package.sh weights/det.onnx                                    # detector-only
#   ./submission/package.sh weights/det.onnx weights/cls.onnx                   # two-stage
#   ./submission/package.sh weights/det.onnx weights/cls.onnx submission-v8     # custom name
#
# Supports .pt and .onnx weight files. Automatically selects the right run.py:
#   - ONNX detector only         → run_onnx_tta.py
#   - ONNX detector + classifier → run_two_stage.py
#   - .pt detector                → run.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DETECTOR="${1:?Usage: $0 <detector> [classifier] [output-name]}"

# Detect if second arg is a weight file or output name
CLASSIFIER=""
OUTPUT_NAME="submission"

if [ -n "${2:-}" ]; then
    if [[ "$2" == *.pt ]] || [[ "$2" == *.onnx ]]; then
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

# Determine which run.py to use
if [[ "$DETECTOR" == *.onnx ]]; then
    if [ -n "$CLASSIFIER" ]; then
        echo "  Pipeline: two-stage ONNX (detector + classifier)"
        cp "$SCRIPT_DIR/run_two_stage.py" "$STAGING/run.py"

        # Build config for two-stage
        cat > "$STAGING/config.json" << EOF
{
  "nc": 356,
  "conf_threshold": 0.05,
  "iou_nms_threshold": 0.5,
  "max_predictions_per_image": 500,
  "detector_file": "detector.onnx",
  "classifier_file": "classifier.onnx",
  "det_imgsz": 1280,
  "cls_imgsz": 224,
  "cls_conf_threshold": 0.3,
  "cls_override_always": false,
  "crop_padding": 0.1,
  "tta_flip": true
}
EOF
    else
        echo "  Pipeline: detector-only ONNX with TTA"
        cp "$SCRIPT_DIR/run_onnx_tta.py" "$STAGING/run.py"

        # Build config for detector-only
        cat > "$STAGING/config.json" << EOF
{
  "nc": 356,
  "conf_threshold": 0.05,
  "iou_nms_threshold": 0.5,
  "max_predictions_per_image": 500,
  "model_file": "detector.onnx",
  "tta_scales": [1280],
  "tta_flip": true,
  "fusion_method": "concat_nms",
  "time_budget": 260
}
EOF
    fi

    # Copy detector
    cp "$DETECTOR" "$STAGING/detector.onnx"
    WEIGHT_SIZE=$(du -m "$STAGING/detector.onnx" | cut -f1)
    echo "  Detector: ${WEIGHT_SIZE} MB (ONNX)"

    # Copy classifier if provided
    if [ -n "$CLASSIFIER" ]; then
        cp "$CLASSIFIER" "$STAGING/classifier.onnx"
        CLS_SIZE=$(du -m "$STAGING/classifier.onnx" | cut -f1)
        echo "  Classifier: ${CLS_SIZE} MB (ONNX)"
    fi
else
    # .pt detector
    echo "  Pipeline: ultralytics .pt"
    cp "$SCRIPT_DIR/run.py" "$STAGING/run.py"
    cp "$SCRIPT_DIR/config.json" "$STAGING/config.json"
    cp "$DETECTOR" "$STAGING/detector.pt"
    WEIGHT_SIZE=$(du -m "$STAGING/detector.pt" | cut -f1)
    echo "  Detector: ${WEIGHT_SIZE} MB (.pt)"

    if [ -n "$CLASSIFIER" ]; then
        cp "$CLASSIFIER" "$STAGING/classifier.pt"
        CLS_SIZE=$(du -m "$STAGING/classifier.pt" | cut -f1)
        echo "  Classifier: ${CLS_SIZE} MB (.pt)"
    fi
fi

# Check total weight size
TOTAL_SIZE=$(du -sm "$STAGING" | cut -f1)
if [ "$TOTAL_SIZE" -gt 420 ]; then
    echo "  [ERROR] Total size ${TOTAL_SIZE} MB exceeds 420 MB limit!"
    exit 1
fi

# Build zip
cd "$STAGING"
rm -f "$OUTPUT_ZIP"
zip -r "$OUTPUT_ZIP" . -x ".*" "__MACOSX/*"

echo ""
echo "=== Validating ==="
python3 "$SCRIPT_DIR/validate_submission.py" "$OUTPUT_ZIP" || true

echo ""
FINAL_SIZE=$(du -m "$OUTPUT_ZIP" | cut -f1)
echo "=== Done ==="
echo "  Output: $OUTPUT_ZIP (${FINAL_SIZE} MB compressed)"
echo ""
echo "  Upload at: https://app.ainm.no"
