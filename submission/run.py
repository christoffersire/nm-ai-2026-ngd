"""
NorgesGruppen Object Detection — Competition Entry Point

Usage: python run.py --input /data/images --output /output/predictions.json
"""
import argparse
import json
import re
import math
from pathlib import Path

import torch
from ultralytics import YOLO


# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "detector.pt"
CONF_THRESHOLD = 0.25
IOU_NMS_THRESHOLD = 0.5
MAX_PREDICTIONS_PER_IMAGE = 500
NC = 356  # categories 0-355


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


def load_model() -> YOLO | None:
    """Load YOLO model with fail-soft behavior."""
    try:
        if not MODEL_PATH.exists():
            print(f"[WARN] Model not found at {MODEL_PATH}")
            return None
        model = YOLO(str(MODEL_PATH))
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None


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

    # Load model (fail-soft: empty predictions if model fails)
    model = load_model()
    if model is None:
        print("[WARN] No model loaded. Writing empty predictions.")
        output_path.write_text("[]")
        return

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
        all_predictions.extend(preds)

    # Write output
    output_path.write_text(json.dumps(all_predictions))
    print(f"Wrote {len(all_predictions)} predictions for {len(image_files)} images")


if __name__ == "__main__":
    main()
