#!/bin/bash
# Chain training: YOLO26x split → winner on all data
# Run on A100 VM

set -euo pipefail
cd ~/nm-ai-2026-ngd

echo "=== Stage 1: YOLO26x on split data ==="
python3 train/train_detector.py \
  --dataset datasets/full-class/data.yaml \
  --model yolo26x.pt \
  --epochs 100 \
  --imgsz 1280 \
  --batch -1 \
  --patience 20 \
  --name full-class-26x-v1 \
  --device 0

echo ""
echo "=== Stage 1 complete. Comparing results ==="

# Extract best mAP50 from each model
YOLO11X_MAP=$(grep 'all' train_yolo11x.log | awk '{print $6}' | sort -rn | head -1)
YOLO26X_MAP=$(grep 'all' runs/detect/full-class-26x-v1/results.csv 2>/dev/null | tail -1 | cut -d',' -f7 || echo "0")

echo "YOLO11x best mAP50: $YOLO11X_MAP"
echo "YOLO26x best mAP50: $YOLO26X_MAP"

# Determine winner (use python for float comparison)
WINNER=$(python3 -c "
m11 = float('$YOLO11X_MAP') if '$YOLO11X_MAP' else 0
m26 = float('$YOLO26X_MAP') if '$YOLO26X_MAP' else 0
print('yolo11x' if m11 >= m26 else 'yolo26x')
print(f'11x={m11:.4f} vs 26x={m26:.4f}')
")
echo "Winner: $WINNER"

WINNER_MODEL=$(echo $WINNER | head -1)

echo ""
echo "=== Stage 2: Retrain $WINNER_MODEL on ALL data ==="

if [ "$WINNER_MODEL" = "yolo11x" ]; then
  MODEL_PT="yolo11x.pt"
  RUN_NAME="full-class-11x-alldata"
else
  MODEL_PT="yolo26x.pt"
  RUN_NAME="full-class-26x-alldata"
fi

python3 train/train_detector.py \
  --dataset datasets/full-class-alldata/data.yaml \
  --model "$MODEL_PT" \
  --epochs 100 \
  --imgsz 1280 \
  --batch -1 \
  --patience 20 \
  --name "$RUN_NAME" \
  --device 0

echo ""
echo "=== All training complete ==="
echo "Winner: $WINNER_MODEL"
echo "All-data model: runs/detect/$RUN_NAME/weights/best.pt"
