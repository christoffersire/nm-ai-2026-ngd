#!/bin/bash
# Overnight training setup — run on each A100 VM
# Fixes corrupt images, applies mislabel fixes, re-prepares dataset, launches training
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RAW_DIR="${PROJECT_DIR}/data/raw"

IMGSZ=${1:-1280}
RUN_NAME="det-11x-v3-fixed-${IMGSZ}"

echo "=== Overnight Training Setup (imgsz=${IMGSZ}, run=${RUN_NAME}) ==="

cd "${PROJECT_DIR}"

# Step 1: Replace corrupted images
echo "[1/4] Fixing corrupted images..."
cp ~/fixed_images/img_00117.jpg "${RAW_DIR}/images/img_00117.jpg"
cp ~/fixed_images/img_00134.jpg "${RAW_DIR}/images/img_00134.jpg"
python3 -c "
from PIL import Image
for f in ['img_00117.jpg', 'img_00134.jpg']:
    img = Image.open(f'${RAW_DIR}/images/{f}')
    img.load()
    print(f'  {f}: OK ({img.size})')
"

# Step 2: Use fixed annotations
echo "[2/4] Using fixed annotations (10 mislabel corrections)..."
cp "${RAW_DIR}/annotations_fixed.json" "${RAW_DIR}/annotations.json.bak"
cp "${RAW_DIR}/annotations_fixed.json" "${RAW_DIR}/annotations.json"
python3 -c "
import json
with open('${RAW_DIR}/annotations.json') as f:
    data = json.load(f)
print(f'  Annotations: {len(data[\"annotations\"])}, Categories: {len(data[\"categories\"])}')
"

# Step 3: Re-prepare YOLO dataset
echo "[3/4] Re-preparing YOLO dataset (all-data)..."
python3 data/scripts/prepare_alldata.py

# Step 4: Launch training
echo "[4/4] Launching training: YOLO11x, imgsz=${IMGSZ}, 300 epochs..."
nohup python3 train/train_detector.py \
    --dataset datasets/full-class-alldata/data.yaml \
    --model yolo11x.pt \
    --epochs 300 \
    --imgsz ${IMGSZ} \
    --batch -1 \
    --patience 50 \
    --name ${RUN_NAME} \
    --device 0 \
    > ~/train_overnight_${IMGSZ}.log 2>&1 &

echo "Training PID: $!"
echo "Log: ~/train_overnight_${IMGSZ}.log"
echo "Monitor: tail -f ~/train_overnight_${IMGSZ}.log"
echo "Done! Training is running in background."
