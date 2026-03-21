# NM i AI 2026 — NorgesGruppen Object Detection

Grocery product detection and classification on store shelf images.

## Scoring

Score = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5

- Detection component: IoU ≥ 0.5, category ignored (max 70%)
- Classification component: IoU ≥ 0.5 + correct category_id (remaining 30%)

## Data

- 248 shelf images, 22,731 annotations, 356 categories (nc=356, ids 0-355)
- 41 categories have only 1 annotation; category 355 = `unknown_product`
- ~4,272 annotations (18.8%) intentionally corrupted by competition organizers
- Product reference images for 327/329 products (multi-angle: front, back, left, right, top, bottom)
- CDN product images: 354 images from bilder.ngdata.no

## Project Structure

```
data/
  raw/                          # Competition-provided data (gitignored)
    images/                     # 248 shelf images
    annotations.json            # Original COCO annotations (corrupted)
    annotations_fixed.json      # Manually fixed (10 corrections)
    product_images/             # Reference images by barcode (329 products)
  metadata.json                 # Product metadata (corrected_count per product)
  category_mapping.json         # Category ID → name/barcode mapping
  val_split.json                # Fixed train/val split (199/49)
  product_images/               # 354 CDN product images
  scripts/
    clean_annotations.py        # Automated mislabel detection + correction
    prepare_yolo.py             # COCO → YOLO format
    prepare_alldata.py          # All-248-images variant (no holdout)
    audit.py                    # Data audit + category mapping
    download_product_images.py  # Download product images from CDN
scoring/
  scorer.py                     # Local hybrid scorer (mirrors competition logic)
  tune_thresholds.py            # Confidence/NMS threshold sweep
  visualize.py                  # Prediction vs GT visualization
  run_stage1.py                 # TTA baseline experiments
  find_mislabels.py             # Detector-based mislabel finder
  tests/test_scorer.py
train/
  train_detector.py             # YOLO detection training
  train_classifier.py           # YOLO classify training (two-stage)
  prepare_classifier_data.py    # Crop extraction + CDN images for classifier
  train_all.sh                  # Orchestrate parallel training across VMs
  overnight_setup.sh            # VM setup for overnight training runs
  eval_two_stage.py             # Two-stage pipeline evaluation
submission/
  run.py                        # Competition entry (ultralytics .pt)
  run_onnx_tta.py               # ONNX + TTA (flip + multi-scale)
  run_two_stage.py              # Two-stage ONNX: detector + classifier + TTA
  run_ensemble.py               # Multi-model WBF ensemble
  validate_submission.py        # Zip structure checker
  package.sh                    # Build submission zip
  config.json                   # Inference configuration
weights/                        # Current best ONNX weights (gitignored)
datasets/                       # Generated YOLO-format datasets (gitignored)
archive/                        # Old submissions and weights (gitignored)
```

## GCP Infrastructure

| VM | Zone | Machine | GPU | Purpose |
|----|------|---------|-----|---------|
| `yolo-trainer` | europe-west1-c | g2-standard-8 | 1x NVIDIA L4 (24GB) | **Smoke testing** — matches competition sandbox |
| `yolo-trainer-a100` | us-central1-f | a2-highgpu-1g | 1x NVIDIA A100 (40GB) | **Training** |
| `yolo-trainer-a100-2` | us-central1-f | a2-highgpu-1g | 1x NVIDIA A100 (40GB) | **Training** |

```bash
gcloud compute ssh yolo-trainer-a100 --zone=us-central1-f --project=ai-nm26osl-1799
gcloud compute ssh yolo-trainer-a100-2 --zone=us-central1-f --project=ai-nm26osl-1799
gcloud compute ssh yolo-trainer --zone=europe-west1-c --project=ai-nm26osl-1799
```

## Workflow

### 1. Data Cleaning

```bash
# Detect mislabeled annotations using trained detector
python data/scripts/clean_annotations.py                # Dry run (report only)
python data/scripts/clean_annotations.py --apply        # Write corrected annotations
python data/scripts/clean_annotations.py --verify       # Visual verification
```

### 2. Data Preparation

```bash
python data/scripts/prepare_yolo.py       # COCO → YOLO format (199 train / 49 val)
python data/scripts/prepare_alldata.py    # All 248 images for final model
```

### 3. Training (on A100 VMs)

```bash
python3 train/train_detector.py \
  --dataset datasets/full-class-alldata/data.yaml \
  --model yolo11x.pt \
  --epochs 300 --imgsz 1280 --batch -1 --patience 30 \
  --name det-11x-v4-clean --device 0
```

### 4. Export to ONNX

```bash
python3 -c "
from ultralytics import YOLO
model = YOLO('runs/detect/det-11x-v4-clean/weights/best.pt')
model.export(format='onnx', imgsz=1280, opset=17, simplify=True, dynamic=True)
"
```

Note: Sandbox uses ultralytics 8.1.0 + PyTorch 2.6.0 which can't load .pt files from newer versions. ONNX export bypasses this incompatibility.

### 5. Package & Submit

```bash
./submission/package.sh weights/det.onnx submission-v14
```

Upload zip at app.ainm.no. Max 420 MB, ≤3 weight files, ≤10 .py files.

**Competition sandbox**: Python 3.11, PyTorch 2.6.0+cu124, ultralytics 8.1.0, onnxruntime-gpu 1.20.0, NVIDIA L4, 300s timeout, no internet.

### Submission Limits

- 6 submissions per day (2 infrastructure freebies), 2 in-flight max
- Timeout: 360s total (300s inference + 60s cold start headroom)

## Current Results

| Submission | Model | Leaderboard | Runtime | Notes |
|-----------|-------|-------------|---------|-------|
| v1 | YOLOv8m (split, .pt) | 0.0 | — | Failed — .pt loading incompatible |
| v2 | YOLOv8m (split, ONNX) | 0.7795 | 24.6s | First working submission |
| v5 | Ensemble (2× medium, WBF) | 0.7860 | 47.6s | Two medium models |
| v7 | YOLO11x (all-data, ONNX, TTA flip) | **0.9019** | 36.9s | Improved aug, 200 epochs |
| v8 | YOLO11x + classifier | 0.8978 | — | Classifier hurt score |
| v13 | Ensemble v3-1280 + v3-1536 (WBF) | **0.9127** | 268s | 10 manual fixes, 300 epochs |
| v14a | Ensemble v4-1280 + v4-1536 (WBF) | pending | ~268s | 1,488 mislabels removed, 150ep |
| v14b | Same + both models flip TTA | pending | ~358s | Tests full TTA within 360s budget |

## Data Cleaning

- **4,272 annotations (18.8%) intentionally corrupted** by competition organizers
- 3-model vision verification (Claude + GPT + Gemini): crop vs reference image, majority vote
- **1,488 confirmed mislabels removed** → 21,241 clean training annotations
- All 356 categories verified against Kassal.app API; 29 EAN codes added
- Reference images for 353/356 categories (only cat 285 Leka Egg missing)

## Key Findings

- **Training recipe matters most**: `close_mosaic=15`, `degrees=5`, `shear=2`, 200+ epochs → +0.116 on leaderboard
- **Data cleaning is the biggest lever**: 10 manual fixes → +0.0108 (v7→v13); 1,488 automated fixes pending
- **Copy-paste augmentation hurts** on dense shelf images
- **Synthetic shelf data hurts** — doesn't match real distribution
- **Classifier stage hurts** — detector's own classification is strong enough
- **ONNX required** — sandbox ultralytics version can't load newer .pt files
- **Flip TTA** is cheap (~2x runtime) and adds ~0.01 mAP
