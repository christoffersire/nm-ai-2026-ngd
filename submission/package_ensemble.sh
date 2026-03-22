#!/bin/bash
# Package a 2- or 3-model ONNX detector ensemble for submission.
#
# Usage:
#   ./submission/package_ensemble.sh <model1> <imgsz1> <flip1> <weight1> \
#                                    <model2> <imgsz2> <flip2> <weight2> \
#                                    [<model3> <imgsz3> <flip3> <weight3>] \
#                                    [output-name]
#
# Example:
#   ./submission/package_ensemble.sh \
#     weights/v3-1280.onnx 1280 true 1.0 \
#     weights/v3-1536-fp16.onnx 1536 false 1.2 \
#     weights/tail-fp16.onnx 1280 false 0.8 \
#     submission-v13-tail

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ "$#" -ne 8 ] && [ "$#" -ne 9 ] && [ "$#" -ne 12 ] && [ "$#" -ne 13 ]; then
  echo "Usage:"
  echo "  $0 <m1> <imgsz1> <flip1> <weight1> <m2> <imgsz2> <flip2> <weight2> [output-name]"
  echo "  $0 <m1> <imgsz1> <flip1> <weight1> <m2> <imgsz2> <flip2> <weight2> <m3> <imgsz3> <flip3> <weight3> [output-name]"
  exit 1
fi

MODEL1="$1"; IMGSZ1="$2"; FLIP1="$3"; WEIGHT1="$4"
MODEL2="$5"; IMGSZ2="$6"; FLIP2="$7"; WEIGHT2="$8"

MODEL3=""
IMGSZ3=""
FLIP3=""
WEIGHT3=""
OUTPUT_NAME="submission-ensemble"

if [ "$#" -eq 9 ]; then
  OUTPUT_NAME="$9"
elif [ "$#" -ge 12 ]; then
  MODEL3="$9"; IMGSZ3="${10}"; FLIP3="${11}"; WEIGHT3="${12}"
  if [ "$#" -eq 13 ]; then
    OUTPUT_NAME="${13}"
  fi
fi

OUTPUT_ZIP="${PROJECT_DIR}/${OUTPUT_NAME}.zip"
STAGING=$(mktemp -d)
trap "rm -rf $STAGING" EXIT

echo "=== Packaging ensemble submission ==="

cp "$SCRIPT_DIR/run_ensemble.py" "$STAGING/run.py"

cp "$MODEL1" "$STAGING/model_a.onnx"
cp "$MODEL2" "$STAGING/model_b.onnx"

MODELS_JSON=$(cat <<EOF
[
  {"name": "A", "file": "model_a.onnx", "imgsz": ${IMGSZ1}, "tta_flip": ${FLIP1}, "weight": ${WEIGHT1}},
  {"name": "B", "file": "model_b.onnx", "imgsz": ${IMGSZ2}, "tta_flip": ${FLIP2}, "weight": ${WEIGHT2}}
EOF
)

ZIP_FILES=("run.py" "config.json" "model_a.onnx" "model_b.onnx")

if [ -n "$MODEL3" ]; then
  cp "$MODEL3" "$STAGING/model_c.onnx"
  MODELS_JSON+=$(cat <<EOF
,
  {"name": "C", "file": "model_c.onnx", "imgsz": ${IMGSZ3}, "tta_flip": ${FLIP3}, "weight": ${WEIGHT3}}
EOF
)
  ZIP_FILES+=("model_c.onnx")
fi

MODELS_JSON+=$'\n]'

cat > "$STAGING/config.json" <<EOF
{
  "nc": 356,
  "conf_threshold": 0.05,
  "max_predictions_per_image": 500,
  "wbf_iou_threshold": 0.55,
  "wbf_skip_threshold": 0.0,
  "models": ${MODELS_JSON}
}
EOF

TOTAL_SIZE=$(du -sm "$STAGING" | cut -f1)
if [ "$TOTAL_SIZE" -gt 420 ]; then
  echo "  [ERROR] Total size ${TOTAL_SIZE} MB exceeds 420 MB limit!"
  exit 1
fi

cd "$STAGING"
rm -f "$OUTPUT_ZIP"
zip -0 "$OUTPUT_ZIP" "${ZIP_FILES[@]}"

echo ""
echo "=== Validating ==="
python3 "$SCRIPT_DIR/validate_submission.py" "$OUTPUT_ZIP" || true

echo ""
FINAL_SIZE=$(du -m "$OUTPUT_ZIP" | cut -f1)
echo "=== Done ==="
echo "  Output: $OUTPUT_ZIP (${FINAL_SIZE} MB compressed)"

