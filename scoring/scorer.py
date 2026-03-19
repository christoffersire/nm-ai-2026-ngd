"""
Local hybrid scorer mirroring competition logic.

Score = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5

Detection mAP: IoU ≥ 0.5, category ignored
Classification mAP: IoU ≥ 0.5, category must match
"""
import json
from pathlib import Path
from collections import defaultdict


def compute_iou(box_a, box_b):
    """Compute IoU between two COCO-format boxes [x, y, w, h]."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix) * max(0, iy2 - iy)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def match_predictions_to_gt(preds, gt_boxes, iou_threshold=0.5, require_class=False):
    """
    Match predictions to GT boxes (one-to-one, greedy by score).

    Returns list of (is_tp, pred_score, pred_cat, gt_cat) tuples.
    Predictions must be sorted by descending score before calling.
    """
    matched_gt = set()
    results = []

    for pred in preds:
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue

            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            gt = gt_boxes[best_gt_idx]
            if require_class:
                is_tp = pred["category_id"] == gt["category_id"]
            else:
                is_tp = True
            matched_gt.add(best_gt_idx)
        else:
            is_tp = False

        results.append((is_tp, pred["score"]))

    return results


def compute_ap(tp_list, n_gt):
    """
    Compute Average Precision from a list of (is_tp, score) tuples.
    Assumes tp_list is sorted by descending score.
    """
    if n_gt == 0:
        return 0.0

    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []

    for is_tp, score in tp_list:
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1

        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / n_gt
        precisions.append(precision)
        recalls.append(recall)

    if not precisions:
        return 0.0

    # COCO-style AP: 101-point interpolation
    import numpy as np
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # 101-point interpolation
    recall_thresholds = np.linspace(0, 1, 101)
    ap = 0
    for t in recall_thresholds:
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max()

    return ap / 101


def compute_map(predictions, ground_truth, require_class=False, iou_threshold=0.5):
    """
    Compute mAP across all images.

    Args:
        predictions: list of prediction dicts with image_id, category_id, bbox, score
        ground_truth: list of GT annotation dicts with image_id, category_id, bbox
        require_class: if True, TP requires matching category_id
        iou_threshold: IoU threshold for matching
    """
    # Group by image
    pred_by_img = defaultdict(list)
    gt_by_img = defaultdict(list)

    for p in predictions:
        pred_by_img[p["image_id"]].append(p)
    for g in ground_truth:
        gt_by_img[g["image_id"]].append(g)

    # Get all image IDs (union of pred and gt)
    all_img_ids = set(pred_by_img.keys()) | set(gt_by_img.keys())

    if require_class:
        # Per-class AP, then mean
        all_cats = set()
        for g in ground_truth:
            all_cats.add(g["category_id"])

        aps = []
        for cat_id in all_cats:
            all_tp_list = []
            total_gt = 0

            for img_id in all_img_ids:
                # Filter preds and gt for this class
                img_preds = [p for p in pred_by_img.get(img_id, []) if p["category_id"] == cat_id]
                img_gt = [g for g in gt_by_img.get(img_id, []) if g["category_id"] == cat_id]

                total_gt += len(img_gt)

                # Sort preds by descending score
                img_preds.sort(key=lambda x: x["score"], reverse=True)

                matches = match_predictions_to_gt(
                    img_preds, img_gt, iou_threshold=iou_threshold, require_class=True
                )
                all_tp_list.extend(matches)

            # Sort all matches by score (across images)
            all_tp_list.sort(key=lambda x: x[1], reverse=True)
            ap = compute_ap(all_tp_list, total_gt)
            aps.append(ap)

        return sum(aps) / len(aps) if aps else 0.0
    else:
        # Category-agnostic: single class AP
        all_tp_list = []
        total_gt = 0

        for img_id in all_img_ids:
            img_preds = pred_by_img.get(img_id, [])
            img_gt = gt_by_img.get(img_id, [])

            total_gt += len(img_gt)

            img_preds_sorted = sorted(img_preds, key=lambda x: x["score"], reverse=True)
            matches = match_predictions_to_gt(
                img_preds_sorted, img_gt, iou_threshold=iou_threshold, require_class=False
            )
            all_tp_list.extend(matches)

        all_tp_list.sort(key=lambda x: x[1], reverse=True)
        return compute_ap(all_tp_list, total_gt)


def hybrid_score(predictions, ground_truth, iou_threshold=0.5):
    """
    Compute hybrid score: 0.7 × detection_mAP + 0.3 × classification_mAP
    """
    det_map = compute_map(predictions, ground_truth, require_class=False, iou_threshold=iou_threshold)
    cls_map = compute_map(predictions, ground_truth, require_class=True, iou_threshold=iou_threshold)
    score = 0.7 * det_map + 0.3 * cls_map

    return {
        "detection_mAP": det_map,
        "classification_mAP": cls_map,
        "hybrid_score": score,
    }


def evaluate_from_files(pred_path: str, gt_path: str, val_ids: list[int] | None = None):
    """Evaluate predictions against ground truth from files."""
    with open(pred_path) as f:
        predictions = json.load(f)

    with open(gt_path) as f:
        gt_data = json.load(f)

    # Extract GT annotations
    ground_truth = gt_data["annotations"]

    # Filter to val set if specified
    if val_ids is not None:
        val_set = set(val_ids)
        predictions = [p for p in predictions if p["image_id"] in val_set]
        ground_truth = [g for g in ground_truth if g["image_id"] in val_set]

    result = hybrid_score(predictions, ground_truth)

    print(f"Detection mAP@0.5:       {result['detection_mAP']:.4f}")
    print(f"Classification mAP@0.5:  {result['classification_mAP']:.4f}")
    print(f"Hybrid Score:            {result['hybrid_score']:.4f}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--val-split", help="Path to val_split.json")
    args = parser.parse_args()

    val_ids = None
    if args.val_split:
        with open(args.val_split) as f:
            val_ids = json.load(f)["val_ids"]

    evaluate_from_files(args.predictions, args.ground_truth, val_ids)
