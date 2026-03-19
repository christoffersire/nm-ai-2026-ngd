"""
NorgesGruppen Object Detection — Competition Entry Point

Two-stage pipeline: YOLOv8 detector → crop classifier → fusion.
Falls back to detector-only if classifier is unavailable.

Usage: python run.py --input /data/images --output /output/predictions.json
"""
import argparse
import json
import re
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from ultralytics import YOLO


# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

# Load config (with defaults)
_DEFAULTS = {
    "nc": 356,
    "conf_threshold": 0.05,
    "iou_nms_threshold": 0.5,
    "max_predictions_per_image": 500,
    "model_file": "detector.pt",
    "classifier_file": "classifier.pt",
    "use_classifier": True,
    "crop_padding": 0.1,
    "classifier_imgsz": 224,
    "classifier_batch_size": 64,
    "fusion_policy": "B",
    "classifier_conf_threshold": 0.5,
    "selective_det_threshold": 0.7,
    "score_formula": "det_score",
}

if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as _f:
        _cfg = json.load(_f)
    _DEFAULTS.update(_cfg)

NC = _DEFAULTS["nc"]
CONF_THRESHOLD = _DEFAULTS["conf_threshold"]
IOU_NMS_THRESHOLD = _DEFAULTS["iou_nms_threshold"]
MAX_PREDICTIONS_PER_IMAGE = _DEFAULTS["max_predictions_per_image"]
MODEL_PATH = SCRIPT_DIR / _DEFAULTS["model_file"]
CLASSIFIER_PATH = SCRIPT_DIR / _DEFAULTS["classifier_file"]
USE_CLASSIFIER = _DEFAULTS["use_classifier"]
CROP_PADDING = _DEFAULTS["crop_padding"]
CLASSIFIER_IMGSZ = _DEFAULTS["classifier_imgsz"]
CLASSIFIER_BATCH_SIZE = _DEFAULTS["classifier_batch_size"]
FUSION_POLICY = _DEFAULTS["fusion_policy"]
CLS_CONF_THRESH = _DEFAULTS["classifier_conf_threshold"]
DET_CONF_THRESH = _DEFAULTS["selective_det_threshold"]
SCORE_FORMULA = _DEFAULTS["score_formula"]


def extract_image_id(filename: str) -> int | None:
    """Extract numeric image ID from filename like img_00042.jpg."""
    m = re.match(r"img_(\d+)\.(jpg|jpeg|png)", filename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def sanitize_predictions(preds: list, img_width: int, img_height: int) -> list:
    """Clamp, filter, and convert predictions to valid format."""
    clean = []
    for p in preds:
        x, y, w, h = p["bbox"]

        # Skip NaN/inf
        if any(math.isnan(v) or math.isinf(v) for v in [x, y, w, h, p["score"]]):
            continue

        # Clamp bbox
        x = max(0.0, x)
        y = max(0.0, y)
        w = max(0.0, min(w, img_width - x))
        h = max(0.0, min(h, img_height - y))

        # Skip zero/negative area
        if w <= 0 or h <= 0:
            continue

        # Clamp score
        score = max(0.0, min(1.0, p["score"]))

        # Clamp category_id
        cat_id = int(p["category_id"])
        if cat_id < 0 or cat_id >= NC:
            cat_id = 0

        clean.append({
            "image_id": int(p["image_id"]),
            "category_id": cat_id,
            "bbox": [float(x), float(y), float(w), float(h)],
            "score": float(score),
        })

    return clean


def load_model(path: Path, name: str = "model") -> YOLO | None:
    """Load YOLO model with fail-soft behavior."""
    try:
        if not path.exists():
            print(f"[WARN] {name} not found at {path}")
            return None
        model = YOLO(str(path))
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load {name}: {e}")
        return None


def crop_with_padding(img, bbox, padding=CROP_PADDING):
    """Crop COCO [x,y,w,h] bbox from PIL image with padding."""
    x, y, w, h = bbox
    img_w, img_h = img.size
    pad_x = w * padding
    pad_y = h * padding
    x1 = max(0, int(x - pad_x))
    y1 = max(0, int(y - pad_y))
    x2 = min(img_w, int(x + w + pad_x))
    y2 = min(img_h, int(y + h + pad_y))
    if x2 <= x1 or y2 <= y1:
        return None
    return img.crop((x1, y1, x2, y2))


def classify_and_fuse(classifier, img, preds):
    """Classify detected crops and apply fusion policy to update predictions."""
    if not preds:
        return preds

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine which detections need classification
    to_classify = []
    indices = []

    for i, p in enumerate(preds):
        needs_cls = True
        if FUSION_POLICY in ("D", "E") and p["score"] >= DET_CONF_THRESH:
            needs_cls = False
        if needs_cls:
            crop = crop_with_padding(img, p["bbox"])
            if crop is not None and crop.size[0] > 2 and crop.size[1] > 2:
                to_classify.append(crop)
                indices.append(i)

    if not to_classify:
        return preds

    # Batch classify with graceful degradation
    cls_results = []
    batch_size = CLASSIFIER_BATCH_SIZE
    for start in range(0, len(to_classify), batch_size):
        batch = to_classify[start:start + batch_size]
        np_batch = [np.array(c.resize((CLASSIFIER_IMGSZ, CLASSIFIER_IMGSZ))) for c in batch]
        try:
            results = classifier.predict(source=np_batch, verbose=False, device=device)
            cls_results.extend(results)
        except Exception:
            # Fallback: smaller batches
            for img_arr in np_batch:
                try:
                    r = classifier.predict(source=img_arr, verbose=False, device=device)
                    cls_results.extend(r)
                except Exception:
                    cls_results.append(None)

    # Apply fusion
    for j, idx in enumerate(indices):
        if j >= len(cls_results) or cls_results[j] is None or cls_results[j].probs is None:
            continue

        cls_top1 = int(cls_results[j].probs.top1)
        cls_conf = float(cls_results[j].probs.top1conf)
        det_class = preds[idx]["category_id"]
        det_conf = preds[idx]["score"]

        # Fusion policy
        final_class = det_class
        final_score = det_conf

        if FUSION_POLICY == "A":
            final_class = cls_top1
            final_score = det_conf

        elif FUSION_POLICY == "B":
            if cls_conf >= CLS_CONF_THRESH:
                final_class = cls_top1
            final_score = det_conf

        elif FUSION_POLICY == "C":
            if cls_conf >= CLS_CONF_THRESH:
                final_class = cls_top1
                final_score = det_conf * cls_conf
            # else keep det score

        elif FUSION_POLICY in ("D", "E"):
            if cls_conf >= CLS_CONF_THRESH and det_conf < DET_CONF_THRESH:
                final_class = cls_top1
            final_score = det_conf

        # Apply score formula
        if SCORE_FORMULA == "det_x_cls":
            final_score = det_conf * cls_conf
        elif SCORE_FORMULA == "det_score":
            final_score = det_conf

        # Clamp
        if final_class < 0 or final_class >= NC:
            final_class = det_class

        preds[idx]["category_id"] = final_class
        preds[idx]["score"] = final_score

    return preds


def predict_image(model: YOLO, image_path: Path, image_id: int) -> list:
    """Run inference on a single image. Returns list of prediction dicts."""
    try:
        results = model.predict(
            source=str(image_path),
            conf=CONF_THRESHOLD,
            iou=IOU_NMS_THRESHOLD,
            max_det=MAX_PREDICTIONS_PER_IMAGE,
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        preds = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            img_h, img_w = result.orig_shape

            for i in range(len(boxes)):
                # YOLO returns xyxy, convert to xywh (COCO format)
                xyxy = boxes.xyxy[i].cpu().tolist()
                x1, y1, x2, y2 = xyxy
                bw = x2 - x1
                bh = y2 - y1

                score = float(boxes.conf[i].cpu())
                cat_id = int(boxes.cls[i].cpu())

                preds.append({
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [x1, y1, bw, bh],
                    "score": score,
                })

            # Sanitize
            preds = sanitize_predictions(preds, img_w, img_h)

        return preds

    except Exception as e:
        print(f"[ERROR] Inference failed for {image_path.name}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to images directory")
    parser.add_argument("--output", required=True, help="Path to output predictions.json")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load detector (fail-soft: empty predictions if model fails)
    model = load_model(MODEL_PATH, "detector")
    if model is None:
        print("[WARN] No detector loaded. Writing empty predictions.")
        output_path.write_text("[]")
        return

    # Load classifier (fail-soft: detector-only if unavailable)
    classifier = None
    if USE_CLASSIFIER:
        classifier = load_model(CLASSIFIER_PATH, "classifier")
        if classifier is None:
            print("[WARN] Classifier not found. Running detector-only mode.")
        else:
            print(f"[INFO] Two-stage mode: policy={FUSION_POLICY}, cls_thresh={CLS_CONF_THRESH}")

    # Set deterministic inference
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Collect image files
    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    all_predictions = []

    for img_path in image_files:
        image_id = extract_image_id(img_path.name)
        if image_id is None:
            print(f"[WARN] Skipping {img_path.name}: cannot extract image ID")
            continue

        preds = predict_image(model, img_path, image_id)

        # Two-stage: classify crops and fuse
        if classifier is not None and preds:
            try:
                img = PILImage.open(img_path).convert("RGB")
                preds = classify_and_fuse(classifier, img, preds)
            except Exception as e:
                print(f"[WARN] Classification failed for {img_path.name}: {e}")

        all_predictions.extend(preds)

    # Write output
    output_path.write_text(json.dumps(all_predictions))
    print(f"Wrote {len(all_predictions)} predictions for {len(image_files)} images")


if __name__ == "__main__":
    main()
