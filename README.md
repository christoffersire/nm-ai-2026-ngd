# NM i AI 2026 — NorgesGruppen Object Detection

Grocery product detection and classification on store shelf images.

## Environment
- Python 3.11
- PyTorch 2.6.0+cu124
- ultralytics 8.1.0
- torchvision 0.21.0+cu124
- timm 0.9.12
- onnxruntime-gpu 1.20.0

## Scoring
Score = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5

## Data
- 248 shelf images, 22,731 annotations, 356 categories (nc=356, ids 0-355)
- Training data: `~/Downloads/train/` (images/ + annotations.json)
- Product reference images: `~/Downloads/NM_NGD_product_images/`

## Submission
```bash
python run.py --input /data/images --output /output/predictions.json
```
