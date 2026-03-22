"""
Evaluate a 2- or 3-model ONNX ensemble on local images using the same
fusion logic as submission/run_ensemble.py.

Examples:
  python3 scoring/eval_ensemble_onnx.py \
    --model weights/v3-1280.onnx --imgsz 1280 --flip true --weight 1.0 \
    --model weights/v3-1536-fp16.onnx --imgsz 1536 --flip false --weight 1.2 \
    --ground-truth data/raw/annotations.json \
    --val-split data/val_split.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "submission"))
sys.path.insert(0, str(PROJECT_DIR / "scoring"))

from run_ensemble import load_sessions, predict_image_ensemble, extract_image_id  # noqa: E402
from scorer import hybrid_score  # noqa: E402


def str_to_bool(value: str):
    value = value.lower().strip()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model path. Repeat 2 or 3 times.",
    )
    parser.add_argument(
        "--imgsz",
        action="append",
        required=True,
        type=int,
        help="Image size per model. Repeat once per model.",
    )
    parser.add_argument(
        "--flip",
        action="append",
        required=True,
        help="Flip TTA per model (true/false). Repeat once per model.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        required=True,
        type=float,
        help="WBF weight per model. Repeat once per model.",
    )
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--images-dir", default=str(PROJECT_DIR / "data" / "raw" / "images"))
    parser.add_argument("--val-split")
    parser.add_argument("--output-predictions")
    parser.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    args = parser.parse_args()

    if not (2 <= len(args.model) <= 3):
        raise SystemExit("Pass 2 or 3 --model arguments")
    if not (len(args.model) == len(args.imgsz) == len(args.flip) == len(args.weight)):
        raise SystemExit("--model/--imgsz/--flip/--weight counts must match")

    model_specs = []
    for idx, model_path in enumerate(args.model):
        model_specs.append(
            {
                "name": chr(ord("A") + idx),
                "path": Path(model_path).resolve(),
                "imgsz": args.imgsz[idx],
                "tta_flip": str_to_bool(args.flip[idx]),
                "weight": args.weight[idx],
            }
        )

    providers = [item.strip() for item in args.providers.split(",") if item.strip()]
    sessions = load_sessions(model_specs, providers)
    if not sessions:
        raise SystemExit("No models could be loaded")

    with open(args.ground_truth) as f:
        gt_data = json.load(f)

    annotations = gt_data["annotations"]
    images = gt_data["images"]
    if args.val_split:
        with open(args.val_split) as f:
            val_ids = set(json.load(f)["val_ids"])
        annotations = [ann for ann in annotations if ann["image_id"] in val_ids]
        images = [img for img in images if img["id"] in val_ids]

    images_dir = Path(args.images_dir).resolve()
    predictions = []

    for img_info in sorted(images, key=lambda row: row["id"]):
        img_path = images_dir / img_info["file_name"]
        if not img_path.exists():
            continue
        image_id = extract_image_id(img_info["file_name"])
        if image_id is None:
            continue

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        preds = predict_image_ensemble(sessions, img_np, img_np.shape[:2])
        for pred in preds:
            pred["image_id"] = image_id
        predictions.extend(preds)

    result = hybrid_score(predictions, annotations)
    print(f"Detection mAP@0.5:       {result['detection_mAP']:.6f}")
    print(f"Classification mAP@0.5:  {result['classification_mAP']:.6f}")
    print(f"Hybrid Score:            {result['hybrid_score']:.6f}")
    print(f"Predictions:             {len(predictions)}")

    if args.output_predictions:
        output_path = Path(args.output_predictions).resolve()
        output_path.write_text(json.dumps(predictions))
        print(f"Wrote predictions to:    {output_path}")


if __name__ == "__main__":
    main()
