"""
Visualize predictions vs ground truth on validation images.

Usage:
  python scoring/visualize.py \
    --model runs/detect/full-class-v1/weights/best.pt \
    --images data/raw/images \
    --ground-truth data/raw/annotations.json \
    --val-split data/val_split.json \
    --output scoring/vis_output \
    --max-images 10

Colors:
  Green  = TP (correct detection + correct class)
  Yellow = detection hit, wrong class
  Red    = false positive (no matching GT)
  Blue   = missed GT (false negative)
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


def compute_iou(box_a, box_b):
    """IoU between two COCO [x, y, w, h] boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0, ix2 - ix) * max(0, iy2 - iy)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_preds_to_gt(preds, gt_boxes, iou_thresh=0.5):
    """
    Match predictions to GT. Returns:
      pred_status: list of ("tp", "wrong_class", "fp") per prediction
      gt_matched: set of matched GT indices
    """
    preds_sorted = sorted(preds, key=lambda p: p["score"], reverse=True)
    gt_matched = set()
    pred_status = []

    for pred in preds_sorted:
        best_iou = 0
        best_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in gt_matched:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx >= 0:
            gt_matched.add(best_idx)
            if pred["category_id"] == gt_boxes[best_idx]["category_id"]:
                pred_status.append("tp")
            else:
                pred_status.append("wrong_class")
        else:
            pred_status.append("fp")

    return preds_sorted, pred_status, gt_matched


def draw_bbox(draw, bbox, color, label, width=2):
    """Draw a COCO [x,y,w,h] bbox with label."""
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
    if label:
        # Background for text
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except (IOError, OSError):
            font = ImageFont.load_default()
        text_bbox = draw.textbbox((x, y), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        draw.rectangle([x, y - th - 4, x + tw + 4, y], fill=color)
        draw.text((x + 2, y - th - 2), label, fill="white", font=font)


def visualize_image(img_path, preds, gt_boxes, cat_names, output_path):
    """Draw predictions and GT on a single image."""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Match predictions to GT
    preds_sorted, pred_status, gt_matched = match_preds_to_gt(preds, gt_boxes)

    # Draw missed GT first (blue, behind everything)
    for i, gt in enumerate(gt_boxes):
        if i not in gt_matched:
            cat_name = cat_names.get(gt["category_id"], str(gt["category_id"]))
            draw_bbox(draw, gt["bbox"], "dodgerblue", f"MISS: {cat_name}", width=2)

    # Draw predictions
    colors = {"tp": "lime", "wrong_class": "yellow", "fp": "red"}

    for pred, status in zip(preds_sorted, pred_status):
        color = colors[status]
        cat_name = cat_names.get(pred["category_id"], str(pred["category_id"]))
        label = f"{cat_name} {pred['score']:.2f}"
        if status == "wrong_class":
            label = f"WRONG: {cat_name} {pred['score']:.2f}"
        elif status == "fp":
            label = f"FP: {cat_name} {pred['score']:.2f}"
        draw_bbox(draw, pred["bbox"], color, label, width=3 if status == "tp" else 2)

    # Summary text
    n_tp = pred_status.count("tp")
    n_wrong = pred_status.count("wrong_class")
    n_fp = pred_status.count("fp")
    n_miss = len(gt_boxes) - len(gt_matched)
    summary = f"TP:{n_tp}  Wrong:{n_wrong}  FP:{n_fp}  Miss:{n_miss}  GT:{len(gt_boxes)}"

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()
    draw.rectangle([0, 0, 500, 24], fill="black")
    draw.text((4, 4), summary, fill="white", font=font)

    img.save(output_path)
    return {"tp": n_tp, "wrong_class": n_wrong, "fp": n_fp, "missed": n_miss, "gt": len(gt_boxes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained .pt model")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--ground-truth", required=True, help="Path to annotations.json")
    parser.add_argument("--val-split", required=True, help="Path to val_split.json")
    parser.add_argument("--output", default="scoring/vis_output", help="Output directory")
    parser.add_argument("--max-images", type=int, default=10, help="Max images to visualize")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    # Load data
    with open(args.ground_truth) as f:
        gt_data = json.load(f)
    with open(args.val_split) as f:
        val_ids = set(json.load(f)["val_ids"])

    # Build category name lookup
    cat_names = {c["id"]: c["name"] for c in gt_data["categories"]}

    # Group GT by image
    gt_by_img = defaultdict(list)
    for ann in gt_data["annotations"]:
        if ann["image_id"] in val_ids:
            gt_by_img[ann["image_id"]].append(ann)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)

    # Find val images
    images_dir = Path(args.images)
    val_images = []
    for f in sorted(images_dir.iterdir()):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        parts = f.stem.split("_")
        try:
            img_id = int(parts[-1])
        except ValueError:
            continue
        if img_id in val_ids:
            val_images.append((f, img_id))

    val_images = val_images[:args.max_images]

    # Output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Visualizing {len(val_images)} val images (conf={args.conf})")
    print()

    totals = {"tp": 0, "wrong_class": 0, "fp": 0, "missed": 0, "gt": 0}

    for img_path, img_id in val_images:
        # Run inference
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=0.5,
            max_det=500,
            verbose=False,
            device=device,
        )

        preds = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().tolist()
                preds.append({
                    "image_id": img_id,
                    "category_id": int(r.boxes.cls[i].cpu()),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(r.boxes.conf[i].cpu()),
                })

        gt_boxes = gt_by_img.get(img_id, [])
        out_path = out_dir / f"vis_{img_path.name}"
        stats = visualize_image(img_path, preds, gt_boxes, cat_names, out_path)

        for k in totals:
            totals[k] += stats[k]

        print(f"  {img_path.name}: TP={stats['tp']} Wrong={stats['wrong_class']} "
              f"FP={stats['fp']} Miss={stats['missed']} (GT={stats['gt']}, Preds={len(preds)})")

    print()
    print(f"Totals: TP={totals['tp']} Wrong={totals['wrong_class']} "
          f"FP={totals['fp']} Miss={totals['missed']} (GT={totals['gt']})")
    print(f"Output saved to {out_dir}/")


if __name__ == "__main__":
    main()
