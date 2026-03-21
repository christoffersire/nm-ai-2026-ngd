#!/bin/bash
# Full training pipeline — run from local machine to orchestrate VMs
#
# Trains detector + classifier in parallel across VMs:
#   VM1 (A100): YOLO11x detector on all data
#   VM2 (L4):   Classifier on crops + CDN images
#
# Prerequisites:
#   - product images downloaded: python data/download_product_images.py
#   - VMs have synced repo: git pull on each VM
#   - Training data at data/raw/ in the project directory on each VM

set -euo pipefail

PROJECT="ai-nm26osl-1799"
A100_VM="yolo-trainer-a100"
A100_ZONE="us-central1-f"
L4_VM="yolo-trainer"
L4_ZONE="europe-west1-c"

echo "============================================"
echo "  NM i AI 2026 — Full Training Pipeline"
echo "============================================"

# --- Step 1: Sync data to VMs ---
echo ""
echo "=== Syncing product images to VMs ==="

# Sync product images
gcloud compute scp --recurse \
  ~/nm-ai-2026-ngd/data/product_images/ \
  ${A100_VM}:~/nm-ai-2026-ngd/data/product_images/ \
  --zone=${A100_ZONE} --project=${PROJECT} &
PID_SYNC_A100=$!

gcloud compute scp --recurse \
  ~/nm-ai-2026-ngd/data/product_images/ \
  ${L4_VM}:~/nm-ai-2026-ngd/data/product_images/ \
  --zone=${L4_ZONE} --project=${PROJECT} &
PID_SYNC_L4=$!

wait $PID_SYNC_A100 $PID_SYNC_L4
echo "Product images synced to both VMs"

# Sync updated scripts
for VM_INFO in "${A100_VM}:${A100_ZONE}" "${L4_VM}:${L4_ZONE}"; do
  VM=$(echo $VM_INFO | cut -d: -f1)
  ZONE=$(echo $VM_INFO | cut -d: -f2)
  gcloud compute scp \
    ~/nm-ai-2026-ngd/train/prepare_classifier_data.py \
    ~/nm-ai-2026-ngd/train/train_detector.py \
    ~/nm-ai-2026-ngd/train/train_classifier.py \
    ~/nm-ai-2026-ngd/data/prepare_yolo.py \
    ${VM}:~/nm-ai-2026-ngd/train/ \
    --zone=${ZONE} --project=${PROJECT}
  # Also sync data scripts
  gcloud compute scp \
    ~/nm-ai-2026-ngd/data/prepare_yolo.py \
    ~/nm-ai-2026-ngd/data/prepare_alldata.py \
    ~/nm-ai-2026-ngd/data/category_mapping.json \
    ~/nm-ai-2026-ngd/data/val_split.json \
    ${VM}:~/nm-ai-2026-ngd/data/ \
    --zone=${ZONE} --project=${PROJECT}
done
echo "Scripts synced"

# --- Step 2: Prepare datasets on VMs ---
echo ""
echo "=== Preparing datasets ==="

# A100: prepare YOLO dataset (all data) + classifier dataset (all data)
gcloud compute ssh ${A100_VM} --zone=${A100_ZONE} --project=${PROJECT} --command="
cd ~/nm-ai-2026-ngd
python3 data/prepare_yolo.py
python3 data/prepare_alldata.py
python3 train/prepare_classifier_data.py --all-data --crop-size 224 --jitter-count 3 --cdn-augments 5
echo 'A100 datasets ready'
" &
PID_PREP_A100=$!

# L4: prepare YOLO dataset (split) + classifier dataset (split, for validation)
gcloud compute ssh ${L4_VM} --zone=${L4_ZONE} --project=${PROJECT} --command="
cd ~/nm-ai-2026-ngd
python3 data/prepare_yolo.py
python3 train/prepare_classifier_data.py --crop-size 224 --jitter-count 3 --cdn-augments 5
echo 'L4 datasets ready'
" &
PID_PREP_L4=$!

wait $PID_PREP_A100 $PID_PREP_L4
echo "Datasets prepared on both VMs"

# --- Step 3: Launch training in parallel ---
echo ""
echo "=== Launching training ==="

# A100: YOLO11x detector (all 248 images, 200 epochs, improved augmentation)
echo "Starting YOLO11x detector training on A100..."
gcloud compute ssh ${A100_VM} --zone=${A100_ZONE} --project=${PROJECT} --command="
cd ~/nm-ai-2026-ngd
nohup python3 train/train_detector.py \
  --dataset datasets/full-class-alldata/data.yaml \
  --model yolo11x.pt \
  --epochs 200 \
  --imgsz 1280 \
  --batch -1 \
  --patience 30 \
  --name det-11x-v2-alldata \
  --device 0 \
  > train_det_11x_v2.log 2>&1 &
echo 'Detector training launched (PID: \$!)'
" &

# L4: Classifier training (on split data with CDN images)
echo "Starting classifier training on L4..."
gcloud compute ssh ${L4_VM} --zone=${L4_ZONE} --project=${PROJECT} --command="
cd ~/nm-ai-2026-ngd
nohup python3 train/train_classifier.py \
  --dataset datasets/classifier \
  --model yolov8m-cls.pt \
  --epochs 100 \
  --imgsz 224 \
  --batch 64 \
  --patience 20 \
  --name cls-v2-cdn \
  --device 0 \
  > train_cls_v2.log 2>&1 &
echo 'Classifier training launched (PID: \$!)'
"

wait
echo ""
echo "=== Training launched on both VMs ==="
echo ""
echo "Monitor with:"
echo "  A100 detector: gcloud compute ssh ${A100_VM} --zone=${A100_ZONE} --project=${PROJECT} --command='tail -f ~/nm-ai-2026-ngd/train_det_11x_v2.log'"
echo "  L4 classifier: gcloud compute ssh ${L4_VM} --zone=${L4_ZONE} --project=${PROJECT} --command='tail -f ~/nm-ai-2026-ngd/train_cls_v2.log'"
echo ""
echo "When done, export to ONNX:"
echo "  python3 -c \"from ultralytics import YOLO; m=YOLO('runs/detect/det-11x-v2-alldata/weights/best.pt'); m.export(format='onnx', imgsz=1280, opset=17, simplify=True, dynamic=True)\""
echo "  python3 -c \"from ultralytics import YOLO; m=YOLO('runs/classify/cls-v2-cdn/weights/best.pt'); m.export(format='onnx', imgsz=224, opset=17, simplify=True)\""
