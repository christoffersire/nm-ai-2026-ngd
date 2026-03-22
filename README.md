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

## Submission History

Our best submission is **v16-full at 0.9134**. Here is the full history showing what worked and what didn't:

### Phase 1: Finding the right architecture (Day 1)

| Submission | Score | What changed | Lesson |
|-----------|-------|-------------|--------|
| v1 | 0.0 | YOLOv8m with .pt weights | Sandbox ultralytics 8.1.0 can't load newer .pt files → must use ONNX |
| v2 | 0.7795 | YOLOv8m exported to ONNX | First working submission |
| v5 | 0.7860 | 2× YOLOv8m ensemble with WBF | Marginal gain from ensemble at this model size |
| v7 | **0.9019** | YOLO11x, improved augmentation, flip TTA | **Huge jump** — training recipe was the key breakthrough |
| v8 | 0.8978 | v7 + separate YOLOv8m classifier | Adding a classifier HURT — detector's built-in classification is better |

**Key insight from Phase 1**: The training recipe (`close_mosaic=15, degrees=5, shear=2, 200 epochs`) was worth +0.116. Architecture (11x vs 8m) and augmentation matter more than inference tricks.

### Phase 2: Data cleaning + ensemble (Day 2)

We discovered the competition data has ~4,272 intentionally corrupted annotations (metadata `corrected_count` field reveals how many per product). We manually verified and fixed 10 mislabeled annotations, replaced 2 corrupted training images, and trained for 300 epochs with a two-model ensemble.

| Submission | Score | What changed | Lesson |
|-----------|-------|-------------|--------|
| **v13** | **0.9127** | Ensemble: YOLO11x 1280px + 1536px, WBF fusion, 10 manual label fixes, 300 epochs | **+0.0108 from just 10 verified label fixes** — data quality is the biggest lever |

**v13 recipe**: Two YOLO11x models (1280px FP32 + 1536px FP16) ensembled via Weighted Box Fusion (WBF). Model weights [1.0, 1.2], WBF IoU=0.55, conf=0.05. Flip TTA on 1280 model only. Trained on all 248 images (no validation holdout), 300 epochs with AdamW, cosine LR.

### Phase 3: Automated data cleaning attempts (Day 3, early)

We built automated pipelines to find and fix the remaining ~4,262 corrupted annotations. All of these made the score WORSE:

| Submission | Score | What changed | Lesson |
|-----------|-------|-------------|--------|
| v14a | 0.8899 | Removed 1,488 "mislabeled" annotations (150ep) | **Removing annotations hurts badly** even if labels are wrong |
| v14b | 0.8926 | Same + flip TTA on both models | Full TTA on both models hurts |
| v14c | 0.8980 | 300ep 1280 + 150ep 1536 | Better but still worse than v13 |
| v14d | 0.8848 | 300ep on both models with cleaned data | Worst result — confirmed removal is destructive |

**Why automated cleaning failed**: Our 3-model vision verification (Claude + GPT + Gemini comparing crops to reference images) had a 35-88% false positive rate. Products have multiple packaging designs (old/new), and the models couldn't distinguish "different packaging of correct product" from "actually wrong product." We ended up removing ~1,200 correct annotations alongside ~300 real errors.

### Phase 4: Targeted fixes with text reading (Day 3, mid)

We switched strategy: instead of comparing images, use Gemini 2.5 Flash to READ THE TEXT on product packaging. This is much more reliable because the text explicitly says the product name.

| Submission | Score | What changed | Lesson |
|-----------|-------|-------------|--------|
| v15a | 0.9113 | v13 exact weights + full TTA both models | Full TTA hurts by -0.0014 (confirmed with identical weights) |
| v16 | 0.9117 | v16-1280 + **v13-1536**, conf=0.01 | Mismatched ensemble: models trained on different data! |
| v16b | 0.9119 | Same but conf=0.05 | Same mismatch — couldn't evaluate data changes fairly |
| v13b | 0.9117 | v13 + multi-scale TTA (1280+960) | Multi-scale TTA hurts by -0.0010 |

**The mismatched ensemble mistake**: For v16/v16b, we retrained only the 1280px model on new data (with 61 unknown_product relabels) but paired it with v13's original 1536px model. The two models disagreed on every relabeled annotation because they learned different labels. This masked the actual improvement from the data changes.

### Phase 5: Matched ensemble breakthrough (Day 3, late)

Once the v16-1536 model finished training (on the same v16 data as the 1280 model), we submitted a properly matched ensemble:

| Submission | Score | What changed | Lesson |
|-----------|-------|-------------|--------|
| **v16-full** | **0.9134** | **v16-1280 + v16-1536, both trained on v16 data** | **New best! Matched ensemble proves data changes help** |

**v16 data changes** (from v13 baseline): 61 unknown_product annotations relabeled to their correct categories (verified by reading product text with Gemini + manual visual review). No annotations removed.

---

## Key Findings (ranked by importance)

1. **Training recipe is the foundation**: `close_mosaic=15, degrees=5, shear=2, mixup=0.15, erasing=0.4, 300 epochs` → +0.116 on leaderboard
2. **Manual label fixes have outsized impact**: 10 fixes → +0.0108 (v7→v13), 61 relabels → +0.0007 (v13→v16-full)
3. **Never remove annotations** — even wrong labels contain useful bounding box data. Removal caused -0.015 to -0.028 regression
4. **Both ensemble models must train on identical data** — mismatched training masks real improvements and can cause regression
5. **Relabel, don't remove** — changing category_id preserves the detection signal while fixing classification
6. **Text reading (Gemini) > image comparison** for finding mislabels — packaging variants cause massive false positives in visual comparison
7. **ONNX export required** — sandbox ultralytics 8.1.0 can't load .pt from newer versions
8. **Full TTA on both models hurts** (-0.0014), multi-scale TTA hurts (-0.0010)
9. **Separate classifier hurts** — detector's built-in class head is better (v8 lesson)
10. **Sandbox is ~3.7x faster than L4 smoke test** (70s vs 262s)

---

## Data Cleaning Pipeline

### The problem
Competition organizers intentionally corrupted ~4,272 annotations by swapping category labels between similar products (e.g., EVERGOOD KOKMALT ↔ FILTERMALT). The metadata file's `corrected_count` field reveals how many annotations per product were tampered with.

### What we tried and learned

**Approach 1: Embedding-based comparison** (FAILED)
- Used ConvNeXt embeddings to compare annotation crops against reference product images
- Problem: matched by visual similarity, not product identity → wrong corrections

**Approach 2: 3-model vision verification** (PARTIALLY WORKED)
- Sent crop + reference image to Claude, GPT, and Gemini → majority vote
- Found real errors but 35-88% false positive rate due to packaging variants (old vs new packaging)

**Approach 3: Gemini text reading** (BEST)
- Ask Gemini 2.5 Flash to read text on the product packaging
- ~57% of crops had readable text; when readable, accuracy was high
- Used for: unknown_product relabeling (61), KOKMALT/FILTERMALT swaps (8), CHEERIOS/CRUESLI swaps (7), small category fixes (3)

### Verified fixes applied

**In v16 (current best, 0.9134):**
- 61 unknown_product → correct category relabels

**In v17 (training):**
- All v16 fixes + 8 KOKMALT→FILTERMALT + 7 CHEERIOS/CRUESLI + 3 small high-corruption fixes
- Total: 79 label changes, 0 removals

### Critical rules for data changes
- NEVER remove annotations — only relabel
- Only accept relabels with explicit text evidence (Gemini reading packaging text)
- Both ensemble models must be retrained on the same modified data
- Manual visual verification required for every change

---

## Infrastructure

| VM | Zone | GPU | Purpose |
|----|------|-----|---------|
| `yolo-trainer` | europe-west1-c | L4 (24GB) | Smoke testing (matches sandbox GPU) |
| `yolo-trainer-a100` | us-central1-f | A100 (40GB) | Training |
| `yolo-trainer-a100-2` | us-central1-f | A100 (40GB) | Training |
| `yolo-trainer-a100-3` | us-central1-f | A100 (40GB) | Training |

GCP project: `ai-nm26osl-1799`. Free compute provided for competition.

```bash
gcloud compute ssh yolo-trainer-a100 --zone=us-central1-f --project=ai-nm26osl-1799
```

---

## Project Structure

```
data/
  raw/                          # Competition data (gitignored)
    images/                     # 248 shelf images
    annotations.json            # Original annotations (corrupted)
    annotations_fixed.json      # v13 base: 10 manual fixes (md5: 396b513b)
    annotations_v16.json        # v16: + 61 unknown relabels
    annotations_v17.json        # v17: + 15 micro-fixes
    product_images/             # Reference images by barcode
  metadata.json                 # Product metadata with corrected_count
  scripts/
    clean_annotations.py        # Detector-based mislabel finder
    clean_embeddings.py         # Embedding-based mislabel finder
    vision_verify.py            # 3-model verification (Claude+GPT+Gemini)
    vision_relabel.py           # 3-model relabeling
    verify_products.py          # Kassal.app API verification
    generate_review_html.py     # Interactive review tool
    prepare_yolo.py             # COCO → YOLO format
    prepare_alldata.py          # All-data variant
scoring/
  scorer.py                     # Local scorer (mirrors competition)
  wbf_sweep.py                  # WBF parameter sweep
train/
  train_detector.py             # YOLO training with tuned augmentation
  overnight_setup.sh            # VM setup for training runs
submission/
  run_ensemble.py               # 2-model WBF ensemble inference
  run_onnx_tta.py               # Single model with multi-scale TTA
  package.sh                    # Build submission zip
```

---

## Reproducing Best Result (v16-full, 0.9134)

```bash
# 1. Start from competition annotations + 10 manual fixes
cp data/raw/annotations_fixed.json data/raw/annotations_v16.json

# 2. Apply 61 unknown_product relabels (from unknown_relabels.json)
python3 -c "apply relabels..."  # See data/raw/annotations_v16.json

# 3. Prepare YOLO dataset
python3 data/scripts/prepare_alldata.py  # Uses annotations_v16.json

# 4. Train BOTH models on same dataset
python3 train/train_detector.py --dataset ... --model yolo11x.pt --epochs 300 --imgsz 1280
python3 train/train_detector.py --dataset ... --model yolo11x.pt --epochs 300 --imgsz 1536

# 5. Export ONNX
# 1280: FP32, dynamic=True
# 1536: FP16 (half=True) to fit in 420MB budget

# 6. Package ensemble with WBF
# config: conf=0.05, wbf_iou=0.55, weights=[1.0, 1.2], tta_flip=[true, false]
```
