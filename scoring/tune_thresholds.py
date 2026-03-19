"""
Sweep confidence and NMS thresholds to find optimal hybrid score.

Usage:
  python scoring/tune_thresholds.py \
    --model runs/detect/full-class-v1/weights/best.pt \
    --dataset datasets/full-class/data.yaml \
    --ground-truth ~/Downloads/train/annotations.json \
    --val-split data/val_split.json

Runs inference at each (conf, nms) pair on validation images,
then scores with the local hybrid scorer. Prints a ranked table.
"""
import argparse
import json
import itertools
from pathlib import Path

import torch
from ultralytics import YOLO

from scorer import hybrid_score


def run_inference(model, image_dir: Path, val_ids: set, conf: float, iou: float) -> list:
    """Run YOLO inference on val images with given thresholds."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions = []

    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    for img_path in image_files:
        # Extract image_id from filename
        stem = img_path.stem
        parts = stem.split("_")
        try:
            image_id = int(parts[-1])
        except ValueError:
            continue

        if image_id not in val_ids:
            continue

        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            max_det=500,
            verbose=False,
            device=device,
        )

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().tolist()
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(r.boxes.cls[i].cpu()),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(r.boxes.conf[i].cpu()),
                })

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained .pt model")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--ground-truth", required=True, help="Path to annotations.json")
    parser.add_argument("--val-split", required=True, help="Path to val_split.json")
    parser.add_argument("--conf-range", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40",
                        help="Comma-separated confidence thresholds to try")
    parser.add_argument("--nms-range", default="0.3,0.4,0.5,0.6,0.7",
                        help="Comma-separated NMS IoU thresholds to try")
    args = parser.parse_args()

    # Load ground truth
    with open(args.ground_truth) as f:
        gt_data = json.load(f)
    ground_truth = gt_data["annotations"]

    # Load val split
    with open(args.val_split) as f:
        val_ids = set(json.load(f)["val_ids"])

    # Filter GT to val
    gt_val = [g for g in ground_truth if g["image_id"] in val_ids]

    # Load model once
    model = YOLO(args.model)

    # Parse threshold ranges
    conf_values = [float(x) for x in args.conf_range.split(",")]
    nms_values = [float(x) for x in args.nms_range.split(",")]

    print(f"Sweeping {len(conf_values)} conf × {len(nms_values)} nms = {len(conf_values) * len(nms_values)} combinations")
    print(f"Val set: {len(val_ids)} images, {len(gt_val)} annotations")
    print()
    print(f"{'conf':>6}  {'nms':>5}  {'det_mAP':>8}  {'cls_mAP':>8}  {'hybrid':>8}  {'n_preds':>8}")
    print("-" * 55)

    results = []

    for conf, nms in itertools.product(conf_values, nms_values):
        preds = run_inference(model, Path(args.images), val_ids, conf, nms)
        scores = hybrid_score(preds, gt_val)

        row = {
            "conf": conf,
            "nms": nms,
            "det_mAP": scores["detection_mAP"],
            "cls_mAP": scores["classification_mAP"],
            "hybrid": scores["hybrid_score"],
            "n_preds": len(preds),
        }
        results.append(row)

        print(f"{conf:6.2f}  {nms:5.2f}  {row['det_mAP']:8.4f}  {row['cls_mAP']:8.4f}  {row['hybrid']:8.4f}  {row['n_preds']:8d}")

    # Sort by hybrid score
    results.sort(key=lambda r: r["hybrid"], reverse=True)

    print()
    print("=== Top 5 configurations ===")
    for i, r in enumerate(results[:5], 1):
        print(f"  {i}. conf={r['conf']:.2f} nms={r['nms']:.2f} → hybrid={r['hybrid']:.4f} "
              f"(det={r['det_mAP']:.4f}, cls={r['cls_mAP']:.4f}, preds={r['n_preds']})")

    # Save full results
    out_path = Path(args.model).parent.parent / "threshold_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
