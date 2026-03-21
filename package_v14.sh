#!/bin/bash
# Package v14 submissions once both models are ready.
#
# Prerequisites:
#   weights/v4-1280.onnx  (150ep, done)
#   weights/v4-1536.onnx  (150ep, from VM2 when finished)
#
# Usage:
#   ./package_v14.sh           # Packages both v14a (safe) and v14b (full TTA)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Check weights exist
if [ ! -f "$SCRIPT_DIR/weights/v4-1280.onnx" ]; then
    echo "ERROR: weights/v4-1280.onnx not found"
    exit 1
fi
if [ ! -f "$SCRIPT_DIR/weights/v4-1536.onnx" ]; then
    echo "ERROR: weights/v4-1536.onnx not found"
    echo "Download with: gcloud compute scp yolo-trainer-a100-2:~/nm-ai-2026-ngd/runs/detect/det-v4-clean-1536-fast*/weights/best.onnx weights/v4-1536.onnx --zone=us-central1-f --project=ai-nm26osl-1799"
    exit 1
fi

echo "=== Packaging v14a (safe: 1280 TTA flip + 1536 no TTA) ==="
rm -rf /tmp/v14a
mkdir -p /tmp/v14a
cp "$SCRIPT_DIR/submission/run_ensemble.py" /tmp/v14a/run.py
cp "$SCRIPT_DIR/weights/v4-1280.onnx" /tmp/v14a/model_a.onnx
cp "$SCRIPT_DIR/weights/v4-1536.onnx" /tmp/v14a/model_b.onnx
cat > /tmp/v14a/config.json << 'EOF'
{
    "nc": 356,
    "conf_threshold": 0.05,
    "model_a": "model_a.onnx",
    "model_b": "model_b.onnx",
    "imgsz_a": 1280,
    "imgsz_b": 1536,
    "wbf_iou_threshold": 0.55,
    "wbf_skip_threshold": 0.0,
    "weights": [1.0, 1.2],
    "tta_flip": true,
    "tta_flip_models": [true, false],
    "max_predictions_per_image": 500
}
EOF
cd /tmp/v14a
rm -f "$SCRIPT_DIR/submission-v14a-safe.zip"
zip -0 "$SCRIPT_DIR/submission-v14a-safe.zip" run.py config.json model_a.onnx model_b.onnx
echo "  Created: submission-v14a-safe.zip ($(du -m "$SCRIPT_DIR/submission-v14a-safe.zip" | cut -f1) MB)"

echo ""
echo "=== Packaging v14b (full TTA: both models flip) ==="
rm -rf /tmp/v14b
mkdir -p /tmp/v14b
cp "$SCRIPT_DIR/submission/run_ensemble.py" /tmp/v14b/run.py
cp "$SCRIPT_DIR/weights/v4-1280.onnx" /tmp/v14b/model_a.onnx
cp "$SCRIPT_DIR/weights/v4-1536.onnx" /tmp/v14b/model_b.onnx
cat > /tmp/v14b/config.json << 'EOF'
{
    "nc": 356,
    "conf_threshold": 0.05,
    "model_a": "model_a.onnx",
    "model_b": "model_b.onnx",
    "imgsz_a": 1280,
    "imgsz_b": 1536,
    "wbf_iou_threshold": 0.55,
    "wbf_skip_threshold": 0.0,
    "weights": [1.0, 1.2],
    "tta_flip": true,
    "tta_flip_models": [true, true],
    "max_predictions_per_image": 500
}
EOF
cd /tmp/v14b
rm -f "$SCRIPT_DIR/submission-v14b-full-tta.zip"
zip -0 "$SCRIPT_DIR/submission-v14b-full-tta.zip" run.py config.json model_a.onnx model_b.onnx
echo "  Created: submission-v14b-full-tta.zip ($(du -m "$SCRIPT_DIR/submission-v14b-full-tta.zip" | cut -f1) MB)"

echo ""
echo "=== Done ==="
echo "Ready to upload at https://app.ainm.no"
echo "  v14a (safe, ~268s):     submission-v14a-safe.zip"
echo "  v14b (full TTA, ~358s): submission-v14b-full-tta.zip"
