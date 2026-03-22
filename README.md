# NM i AI 2026 — NorgesGruppen Object Detection

## Competition Overview

Norwegian AI Championship (NM i AI) 2026, March 19-22. The NorgesGruppen Data task: detect and classify grocery products on store shelf images.

**Scoring**: `Score = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5`
- Detection (70% of score): locate products with bounding boxes (IoU ≥ 0.5, category ignored)
- Classification (30% of score): identify the correct product (IoU ≥ 0.5 + matching category_id)

**Dataset**: 248 shelf images, ~22,700 bounding box annotations, 356 product categories (ids 0-355). Category 355 = `unknown_product`. Competition organizers intentionally corrupted ~4,272 annotations (18.8%) by swapping labels between similar products.

**Sandbox**: NVIDIA L4 (24GB), 360s timeout (300s inference + 60s cold start), no internet. Python 3.11, PyTorch 2.6.0+cu124, ultralytics 8.1.0, onnxruntime-gpu 1.20.0. Max 420MB submission, ≤3 weight files.

**Final ranking** uses a private test set. You can manually select which submission to use for final evaluation.

---

## Final Result

**Best submission: v16-full at 0.9134** — selected for final evaluation.

Two YOLO11x detectors (1280px + 1536px) ensembled via Weighted Box Fusion, both trained on the same cleaned dataset (original competition data + 10 manual label fixes + 61 unknown_product relabels). 300 epochs, all-data training.

---

## Complete Submission History

We made 16 submissions across 3 days. Here is every submission with what changed and what we learned:

### Day 1: Finding the right architecture

| Submission | Score | Runtime | What changed | Lesson |
|-----------|-------|---------|-------------|--------|
| v1 | 0.0 | — | YOLOv8m with .pt weights | Sandbox ultralytics 8.1.0 can't load newer .pt → must use ONNX |
| v2 | 0.7795 | 24.6s | YOLOv8m exported to ONNX | First working submission |
| v5 | 0.7860 | 47.6s | 2× YOLOv8m ensemble with WBF | Marginal gain from ensemble at this model size |
| v7 | **0.9019** | 36.9s | YOLO11x, improved augmentation, flip TTA | **Huge jump** — training recipe was the key (+0.116) |
| v8 | 0.8978 | — | v7 + separate YOLOv8m classifier | Classifier HURT — detector's built-in classification is better |

### Day 2: Data cleaning + ensemble

| Submission | Score | Runtime | What changed | Lesson |
|-----------|-------|---------|-------------|--------|
| **v13** | **0.9127** | 70.2s | 2× YOLO11x ensemble (1280+1536), 10 manual fixes, 300ep | **+0.0108 from 10 verified label fixes** |

### Day 3: Iteration and discovery

| Submission | Score | Runtime | What changed | Lesson |
|-----------|-------|---------|-------------|--------|
| v14a | 0.8899 | 70.8s | Removed 1,488 "mislabeled" annotations | **Removing annotations hurts badly** (-0.023) |
| v14b | 0.8926 | 98.1s | Same + flip TTA on both models | Full TTA both hurts |
| v14c | 0.8980 | 71.6s | 300ep 1280 + 150ep 1536 | Still worse than v13 |
| v14d | 0.8848 | 68.9s | 300ep both with cleaned data | Worst — removal is destructive |
| v15a | 0.9113 | 95.8s | v13 exact weights + full TTA both | Full TTA hurts by -0.0014 (same weights, isolated test) |
| v16 | 0.9117 | 73.3s | v16-1280 + v13-1536, conf=0.01 | **Mismatched ensemble** — models on different data |
| v16b | 0.9119 | 69.4s | Same but conf=0.05 | Same mismatch — couldn't evaluate data changes fairly |
| v13b | 0.9117 | 126.6s | v13 + multi-scale TTA (1280+960) | Multi-scale TTA hurts by -0.0010 |
| **v16-full** | **0.9134** | ~70s | **v16-1280 + v16-1536 (both on same data)** | **New best! Matched ensemble proves relabels help** |
| v16-3model | 0.9112 | ~92s | 3-model ensemble (v16-1280 + v16-1536 + v17-1280) | 3rd model with different data added noise, hurt score |

### The mismatched ensemble mistake

Our most costly error: for v16/v16b, we retrained only the 1280px model on new data (with 61 unknown_product relabels) but paired it with v13's original 1536px model. The two models disagreed on every relabeled annotation because they learned different labels. This made it look like the data changes hurt, when in fact they helped — we just couldn't see it until both models were trained on the same data (v16-full).

---

## What Worked (ranked by impact)

1. **Training recipe**: `close_mosaic=15, degrees=5, shear=2, mixup=0.15, erasing=0.4, 300 epochs, AdamW, cosine LR` → +0.116 improvement (v2→v7)
2. **Multi-scale ensemble**: Two YOLO11x models at 1280px + 1536px with WBF fusion → significant improvement over single model
3. **Manual label fixes**: 10 verified corrections → +0.0108 (v7→v13)
4. **Unknown product relabeling**: 61 annotations relabeled from unknown_product to correct categories → +0.0007 (v13→v16-full), but only measurable with matched ensemble
5. **ONNX export**: Required for sandbox compatibility. Export with `dynamic=True` for flexibility

## What Didn't Work

1. **Removing annotations** → -0.015 to -0.028. Even wrong labels contain useful bounding box data for detection (70% of score). Never remove, only relabel.
2. **Automated data cleaning** with 3-model vision verification (Claude+GPT+Gemini) → 35-88% false positive rate due to packaging variants (old/new packaging designs of same product)
3. **Embedding-based relabeling** → matched by visual similarity, not product identity. Suggested wrong corrections.
4. **Separate classifier stage** → -0.004 (v8). Detector's built-in class head is better.
5. **Full TTA on both models** → -0.0014 (v15a). Only flip TTA on the 1280 model.
6. **Multi-scale TTA (1280+960)** → -0.0010 (v13b). Extra scale added noise.
7. **Lower confidence threshold (0.01 vs 0.05)** → slight negative impact.
8. **3-model ensemble with mismatched data** → -0.002 (v16-3model). Third model trained on different data added noise.
9. **Copy-paste augmentation** on dense shelf images → hurt performance.
10. **Synthetic shelf data generation** → didn't match real distribution.

---

## Data Cleaning: The Full Story

### The problem
Competition organizers intentionally corrupted ~4,272 of 22,731 annotations (18.8%) by swapping category labels between similar products. The metadata file's `corrected_count` field reveals how many annotations per product were tampered with.

### Approach 1: Embedding comparison (FAILED)
Used ConvNeXt embeddings to compare annotation crops against reference product images. Matched by visual similarity → suggested wrong corrections (e.g., Leksands Rutbit → Lano Soap because packaging colors are similar).

### Approach 2: 3-model vision verification (PARTIALLY WORKED)
Sent each annotation's crop + reference image to Claude Haiku, GPT-4o-mini, and Gemini 2.5 Flash simultaneously. Majority vote determined match/mismatch. Found real errors but suffered 35-88% false positive rate because products have multiple packaging designs (old vs new). The models couldn't distinguish "different packaging of correct product" from "actually wrong product."

Manual audit of 100 random samples revealed:
- Category A (models say correct): 97% actually correct ✓
- Category B (one model disagrees): 88% actually correct — mostly packaging variants ✗
- Category C (all 3 say wrong): 61% actually wrong, 39% packaging variant false positives

### Approach 3: Gemini text reading (BEST)
Instead of comparing images, asked Gemini 2.5 Flash to READ the text on product packaging and compare to the label. When Gemini could read text (~44% of crops), accuracy was high. Used for:
- 61 unknown_product relabels (manually verified via text + visual review)
- 8 KOKMALT→FILTERMALT fixes (text explicitly says different product name)
- 7 CHEERIOS MULTI↔HAVRE fixes (text says variant name)
- 3 small high-corruption category fixes (text contradicts label)

### Critical discovery: matched ensemble
When testing data changes, BOTH ensemble models must be trained on the same modified data. Training only one model on new data while keeping the other on old data causes the models to disagree on changed annotations, masking improvements and sometimes causing regression. This mistake cost us 4 submissions (v16, v16b, v16-3model) before we identified it.

### Rules we learned
- **Never remove annotations** — only relabel with explicit text evidence
- **Both ensemble models must train on identical data**
- **Only trust Gemini when it reports actual visible_text** — empty text = unreliable verdict
- **Manual visual verification required** for every change — Gemini has ~43% false positive rate even on text-reading
- **Small categories with high corruption rates** (2-8 annotations, 100% corrupt) are highest impact per fix

---

## Technical Details

### Best submission recipe (v16-full)

**Data**: `annotations_fixed.json` (original competition data + 10 manual label fixes) + 61 unknown_product relabels = 22,731 annotations, 0 removals.

**Training**: YOLO11x, 300 epochs, all 248 images (no validation holdout). AdamW optimizer, lr0=0.001, cosine LR schedule, 3 epoch warmup. Augmentation: mosaic=1.0, mixup=0.15, scale=0.5, fliplr=0.5, degrees=5, translate=0.1, shear=2, hsv_h=0.015, hsv_s=0.5, hsv_v=0.3, erasing=0.4, close_mosaic=15.

**Two models trained on identical data**:
- Model A: imgsz=1280, exported as FP32 ONNX (219MB)
- Model B: imgsz=1536, exported as FP16 ONNX with `half=True` (110MB)

**Inference**: WBF ensemble. conf_threshold=0.05, wbf_iou_threshold=0.55, weights=[1.0, 1.2], flip TTA on model A only. Max 500 predictions per image.

**Runtime**: ~70s on competition sandbox (NVIDIA L4). Smoke tested at ~4m30s on our L4 VM (3.7x slower than sandbox).

### Infrastructure

| VM | Zone | GPU | Purpose |
|----|------|-----|---------|
| `yolo-trainer` | europe-west1-c | L4 (24GB) | Smoke testing (matches sandbox) |
| `yolo-trainer-a100` | us-central1-f | A100 (40GB) | Training |
| `yolo-trainer-a100-2` | us-central1-f | A100 (40GB) | Training |
| `yolo-trainer-a100-3` | us-central1-f | A100 (40GB) | Training |

GCP project: `ai-nm26osl-1799`. Free compute provided for competition.

### Project structure

```
data/
  raw/                          # Competition data (gitignored)
    annotations.json            # Original (corrupted)
    annotations_fixed.json      # v13 base: 10 fixes (md5: 396b513b)
    annotations_v16.json        # v16: + 61 unknown relabels
    annotations_v17.json        # v17: + 15 micro-fixes
    product_images/             # Reference images by barcode
  metadata.json                 # Product metadata with corrected_count
  scripts/
    vision_verify.py            # 3-model verification pipeline
    vision_relabel.py           # 3-model relabeling pipeline
    verify_products.py          # Kassal.app API verification
    generate_review_html.py     # Interactive review tool
    prepare_yolo.py             # COCO → YOLO format
scoring/
  scorer.py                     # Local scorer (mirrors competition)
  wbf_sweep.py                  # WBF parameter sweep
train/
  train_detector.py             # YOLO training with tuned augmentation
submission/
  run_ensemble.py               # N-model WBF ensemble inference
  package.sh                    # Build submission zip
```
