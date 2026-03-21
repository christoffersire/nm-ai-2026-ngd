"""
NorgesGruppen Object Detection — Native ultralytics .pt Entry Point

Uses ultralytics 8.1.0 native inference for optimal NMS and optional TTA.
Requires sandbox to have ultralytics==8.1.0 and torch==2.6.0.

Usage: python run.py --input /data/images --output /output/predictions.json
"""
import argparse
import json
import re
from pathlib import Path

import torch
from ultralytics import YOLO


# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

_DEFAULTS = {
    "nc": 356,
    "conf_threshold": 0.05,
    "iou_nms_threshold": 0.5,
    "max_predictions_per_image": 500,
    "model_file": "detector.pt",
    "use_tta": False,
}

if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as _f:
        _cfg = json.load(_f)
    _DEFAULTS.update(_cfg)

NC = _DEFAULTS["nc"]
CONF_THRESHOLD = _DEFAULTS["conf_threshold"]
IOU_NMS_THRESHOLD = _DEFAULTS["iou_nms_threshold"]
MAX_PREDS = _DEFAULTS["max_predictions_per_image"]
MODEL_PATH = SCRIPT_DIR / _DEFAULTS["model_file"]
USE_TTA = _DEFAULTS["use_tta"]


def extract_image_id(filename):
    m = re.match(r"img_(\d+)\.(jpg|jpeg|png)", filename, re.IGNORECASE)
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    if not MODEL_PATH.exists():
        print(f"[WARN] Model not found at {MODEL_PATH}")
        output_path.write_text("[]")
        return

    try:
        model = YOLO(str(MODEL_PATH))
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        output_path.write_text("[]")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {MODEL_PATH.name}, device: {device}, TTA: {USE_TTA}")

    # Set deterministic
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    all_predictions = []

    for img_path in image_files:
        image_id = extract_image_id(img_path.name)
        if image_id is None:
            continue

        try:
            results = model.predict(
                source=str(img_path),
                conf=CONF_THRESHOLD,
                iou=IOU_NMS_THRESHOLD,
                max_det=MAX_PREDS,
                verbose=False,
                device=device,
                augment=USE_TTA,
            )

            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                for i in range(len(r.boxes)):
                    x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().tolist()
                    w = x2 - x1
                    h = y2 - y1
                    if w <= 0 or h <= 0:
                        continue
                    cat_id = int(r.boxes.cls[i].cpu())
                    if cat_id < 0 or cat_id >= NC:
                        cat_id = 0
                    all_predictions.append({
                        "image_id": image_id,
                        "category_id": cat_id,
                        "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                        "score": round(float(r.boxes.conf[i].cpu()), 4),
                    })
        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")
            continue

    output_path.write_text(json.dumps(all_predictions))
    print(f"Wrote {len(all_predictions)} predictions for {len(image_files)} images")


if __name__ == "__main__":
    main()
