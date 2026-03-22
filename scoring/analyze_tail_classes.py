"""
Analyze per-class classification AP and tail-bucket performance.

This focuses on the macro-classification component that often decides
whether a candidate can beat the current best public submission.

Examples:
  python3 scoring/analyze_tail_classes.py \
    --predictions /tmp/preds.json \
    --ground-truth data/raw/annotations.json \
    --val-split data/val_split.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PROJECT_DIR / "scoring"))
from scorer import compute_ap, hybrid_score, match_predictions_to_gt  # noqa: E402


def load_predictions(path: Path):
    with open(path) as f:
        return json.load(f)


def load_ground_truth(path: Path):
    with open(path) as f:
        data = json.load(f)
    return data


def apply_val_filter(predictions, gt_data, val_split_path: Path | None):
    if val_split_path is None:
        return predictions, gt_data["annotations"], gt_data

    with open(val_split_path) as f:
        val_ids = set(json.load(f)["val_ids"])

    predictions = [p for p in predictions if p["image_id"] in val_ids]
    annotations = [a for a in gt_data["annotations"] if a["image_id"] in val_ids]
    images = [img for img in gt_data["images"] if img["id"] in val_ids]
    filtered = dict(gt_data)
    filtered["annotations"] = annotations
    filtered["images"] = images
    return predictions, annotations, filtered


def compute_per_class_classification_ap(predictions, annotations):
    pred_by_img = defaultdict(list)
    gt_by_img = defaultdict(list)

    for pred in predictions:
        pred_by_img[pred["image_id"]].append(pred)
    for ann in annotations:
        gt_by_img[ann["image_id"]].append(ann)

    cat_ids = sorted({ann["category_id"] for ann in annotations})
    all_img_ids = sorted(set(pred_by_img.keys()) | set(gt_by_img.keys()))

    per_class = []
    for cat_id in cat_ids:
        matches = []
        total_gt = 0

        for img_id in all_img_ids:
            img_preds = [p for p in pred_by_img.get(img_id, []) if p["category_id"] == cat_id]
            img_gt = [g for g in gt_by_img.get(img_id, []) if g["category_id"] == cat_id]
            total_gt += len(img_gt)
            img_preds.sort(key=lambda row: row["score"], reverse=True)
            matches.extend(
                match_predictions_to_gt(
                    img_preds,
                    img_gt,
                    iou_threshold=0.5,
                    require_class=True,
                )
            )

        matches.sort(key=lambda row: row[1], reverse=True)
        ap = compute_ap(matches, total_gt)
        per_class.append(
            {
                "category_id": cat_id,
                "gt_count": total_gt,
                "classification_ap": ap,
            }
        )

    return per_class


def bucket_name(gt_count: int):
    if gt_count == 1:
        return "1"
    if gt_count <= 3:
        return "2-3"
    if gt_count <= 5:
        return "4-5"
    if gt_count <= 10:
        return "6-10"
    if gt_count <= 25:
        return "11-25"
    return "26+"


def summarize_buckets(per_class):
    buckets = defaultdict(list)
    for row in per_class:
        buckets[bucket_name(row["gt_count"])].append(row)

    ordered = ["1", "2-3", "4-5", "6-10", "11-25", "26+"]
    summary = []
    for name in ordered:
        rows = buckets.get(name, [])
        if not rows:
            continue
        summary.append(
            {
                "bucket": name,
                "classes": len(rows),
                "mean_classification_ap": sum(r["classification_ap"] for r in rows) / len(rows),
                "mean_gt_count": sum(r["gt_count"] for r in rows) / len(rows),
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--val-split", help="Optional val_split.json path")
    parser.add_argument("--worst-k", type=int, default=25)
    parser.add_argument("--best-k", type=int, default=10)
    args = parser.parse_args()

    predictions = load_predictions(Path(args.predictions))
    gt_data = load_ground_truth(Path(args.ground_truth))
    val_split_path = Path(args.val_split) if args.val_split else None

    predictions, annotations, gt_data = apply_val_filter(predictions, gt_data, val_split_path)
    category_names = {c["id"]: c["name"] for c in gt_data["categories"]}

    score = hybrid_score(predictions, annotations)
    per_class = compute_per_class_classification_ap(predictions, annotations)
    buckets = summarize_buckets(per_class)

    print("Overall")
    print(f"  Detection mAP@0.5:      {score['detection_mAP']:.6f}")
    print(f"  Classification mAP@0.5: {score['classification_mAP']:.6f}")
    print(f"  Hybrid score:           {score['hybrid_score']:.6f}")
    print()

    print("Tail Buckets")
    for row in buckets:
        print(
            f"  gt {row['bucket']:>5}: {row['classes']:>3} classes | "
            f"mean cls AP {row['mean_classification_ap']:.6f} | "
            f"mean gt {row['mean_gt_count']:.2f}"
        )
    print()

    enriched = []
    for row in per_class:
        enriched.append(
            {
                **row,
                "name": category_names.get(row["category_id"], f"cat_{row['category_id']}"),
            }
        )

    worst = sorted(enriched, key=lambda row: (row["classification_ap"], row["gt_count"], row["category_id"]))
    best = sorted(enriched, key=lambda row: (-row["classification_ap"], row["gt_count"], row["category_id"]))

    print(f"Worst {args.worst_k} Classes")
    for row in worst[:args.worst_k]:
        print(
            f"  cat {row['category_id']:>3} | gt {row['gt_count']:>3} | "
            f"AP {row['classification_ap']:.6f} | {row['name']}"
        )
    print()

    print(f"Best {args.best_k} Classes")
    for row in best[:args.best_k]:
        print(
            f"  cat {row['category_id']:>3} | gt {row['gt_count']:>3} | "
            f"AP {row['classification_ap']:.6f} | {row['name']}"
        )


if __name__ == "__main__":
    main()
