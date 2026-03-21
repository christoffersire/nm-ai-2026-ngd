"""
Evaluate two-stage detection + classification pipeline.

Runs detector on val images, crops detections, classifies with crop classifier,
applies multiple fusion policies, and scores with the hybrid scorer.

Reports: hybrid scores, top-1/top-3 crop accuracy, per-category analysis,
and runtime measurements.

Usage:
  python train/eval_two_stage.py \
    --detector runs/detect/full-class-v1/weights/best.pt \
    --classifier runs/classify/classifier-s-v1/weights/best.pt \
    --images data/raw/images \
    --ground-truth data/raw/annotations.json \
    --val-split data/val_split.json
"""
import argparse
import json
import time
import itertools
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Add parent dir to path for scorer import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scoring"))
from scorer import hybrid_score, compute_map


# ---- Crop Extraction ----

def crop_with_padding(img, bbox, padding=0.1):
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


def compute_iou(box_a, box_b):
    """IoU between two COCO [x,y,w,h] boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0, ix2 - ix) * max(0, iy2 - iy)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ---- Detector Inference ----

def run_detector(model, image_path, image_id, conf=0.05, iou=0.5, max_det=500):
    """Run detector on one image. Returns list of detection dicts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        results = model.predict(
            source=str(image_path), conf=conf, iou=iou,
            max_det=max_det, verbose=False, device=device,
        )
    except Exception:
        return []

    preds = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().tolist()
            preds.append({
                "image_id": image_id,
                "category_id": int(r.boxes.cls[i].cpu()),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(r.boxes.conf[i].cpu()),
            })
    return preds


# ---- Classifier Inference ----

def classify_crops(classifier, crops, imgsz=224, batch_size=64):
    """
    Classify a list of PIL crops. Returns list of (top1_class, top1_conf, top3_classes, top3_confs).
    Falls back to smaller batches on failure.
    """
    if not crops:
        return []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_list = []

    # Process in batches
    for start in range(0, len(crops), batch_size):
        batch = crops[start:start + batch_size]
        # Convert PIL to numpy for ultralytics
        np_batch = [np.array(c.resize((imgsz, imgsz))) for c in batch]

        try:
            results = classifier.predict(
                source=np_batch, verbose=False, device=device,
            )
        except Exception:
            # Fallback: one by one
            results = []
            for img in np_batch:
                try:
                    r = classifier.predict(source=img, verbose=False, device=device)
                    results.extend(r)
                except Exception:
                    results.append(None)

        for r in results:
            if r is None or r.probs is None:
                results_list.append((0, 0.0, [0], [0.0]))
                continue

            top1 = int(r.probs.top1)
            top1conf = float(r.probs.top1conf)

            # Top-3
            top5_indices = r.probs.top5
            top5_confs = r.probs.top5conf
            top3_classes = [int(x) for x in top5_indices[:3]]
            top3_confs = [float(x) for x in top5_confs[:3]]

            results_list.append((top1, top1conf, top3_classes, top3_confs))

    return results_list


# ---- Fusion Policies ----

def apply_fusion(det_class, det_conf, cls_top1, cls_conf, policy, cls_thresh=0.5,
                 det_thresh=0.7, confusion_set=None):
    """
    Apply fusion policy. Returns (final_class, final_score).
    """
    if policy == "A":
        # Always take classifier, keep detector score
        return cls_top1, det_conf

    elif policy == "B":
        # Classifier if confident, else detector
        if cls_conf >= cls_thresh:
            return cls_top1, det_conf
        return det_class, det_conf

    elif policy == "C":
        # Like B but score = det_conf * cls_conf
        if cls_conf >= cls_thresh:
            return cls_top1, det_conf * cls_conf
        return det_class, det_conf

    elif policy == "D":
        # Override only when classifier strong AND detector weak
        if cls_conf >= cls_thresh and det_conf < det_thresh:
            return cls_top1, det_conf
        return det_class, det_conf

    elif policy == "E":
        # Selective: only classify uncertain detections
        if det_conf < det_thresh:
            if cls_conf >= cls_thresh:
                return cls_top1, det_conf
            return det_class, det_conf
        # High-confidence detector: keep as-is
        return det_class, det_conf

    else:
        return det_class, det_conf


# ---- Crop-Level Evaluation ----

def evaluate_crop_accuracy(detections, gt_by_img, classifier, padding=0.1, imgsz=224):
    """
    Evaluate classifier accuracy on detector-matched crops.
    Returns (gt_crop_results, det_crop_results) each as list of (gt_class, det_class, cls_top1, cls_top3).
    """
    # Match detections to GT
    matched_pairs = []  # (det_bbox, gt_class, det_class, det_conf, image_path)

    det_by_img = defaultdict(list)
    for d in detections:
        det_by_img[d["image_id"]].append(d)

    for img_id in det_by_img:
        img_dets = sorted(det_by_img[img_id], key=lambda x: x["score"], reverse=True)
        img_gt = gt_by_img.get(img_id, [])

        matched_gt = set()
        for det in img_dets:
            best_iou = 0
            best_idx = -1
            for gi, gt in enumerate(img_gt):
                if gi in matched_gt:
                    continue
                iou = compute_iou(det["bbox"], gt["bbox"])
                if iou >= 0.5 and iou > best_iou:
                    best_iou = iou
                    best_idx = gi

            if best_idx >= 0:
                matched_gt.add(best_idx)
                matched_pairs.append({
                    "det_bbox": det["bbox"],
                    "gt_class": img_gt[best_idx]["category_id"],
                    "det_class": det["category_id"],
                    "det_conf": det["score"],
                    "image_id": img_id,
                    "image_path": det.get("_image_path"),
                })

    return matched_pairs


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", required=True, help="Detector weights path")
    parser.add_argument("--classifier", required=True, help="Classifier weights path")
    parser.add_argument("--images", required=True, help="Images directory")
    parser.add_argument("--ground-truth", required=True, help="annotations.json path")
    parser.add_argument("--val-split", required=True, help="val_split.json path")
    parser.add_argument("--padding", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=224, help="Classifier input size")
    parser.add_argument("--batch-size", type=int, default=64, help="Classifier batch size")
    args = parser.parse_args()

    # Load data
    with open(args.ground_truth) as f:
        gt_data = json.load(f)
    with open(args.val_split) as f:
        val_ids = set(json.load(f)["val_ids"])

    gt_val = [g for g in gt_data["annotations"] if g["image_id"] in val_ids]
    gt_by_img = defaultdict(list)
    for g in gt_val:
        gt_by_img[g["image_id"]].append(g)

    # Load models
    print("Loading models...")
    detector = YOLO(args.detector)
    classifier = YOLO(args.classifier)

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

    print(f"Val: {len(val_images)} images, {len(gt_val)} annotations")

    # ---- Stage 1: Run detector on all val images ----
    print("\n=== Stage 1: Detection ===")
    all_detections = []
    image_timings = []

    for img_path, img_id in val_images:
        t0 = time.time()
        dets = run_detector(detector, img_path, img_id)
        det_time = time.time() - t0
        for d in dets:
            d["_image_path"] = str(img_path)
        all_detections.extend(dets)
        image_timings.append({"image_id": img_id, "det_time": det_time, "n_dets": len(dets)})

    print(f"  Total detections: {len(all_detections)}")

    # ---- Detector-only baseline ----
    det_only_preds = [{"image_id": d["image_id"], "category_id": d["category_id"],
                       "bbox": d["bbox"], "score": d["score"]} for d in all_detections]
    baseline = hybrid_score(det_only_preds, gt_val)
    print(f"\n=== Detector-Only Baseline ===")
    print(f"  det_mAP:  {baseline['detection_mAP']:.4f}")
    print(f"  cls_mAP:  {baseline['classification_mAP']:.4f}")
    print(f"  hybrid:   {baseline['hybrid_score']:.4f}")

    # ---- Stage 2: Classify crops ----
    print("\n=== Stage 2: Classification ===")

    # Group detections by image for batch cropping
    dets_by_img = defaultdict(list)
    for i, d in enumerate(all_detections):
        dets_by_img[d["image_id"]].append((i, d))

    cls_results = [None] * len(all_detections)

    for img_path, img_id in val_images:
        img_dets = dets_by_img.get(img_id, [])
        if not img_dets:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        t0 = time.time()
        crops = []
        indices = []
        for idx, det in img_dets:
            crop = crop_with_padding(img, det["bbox"], padding=args.padding)
            if crop is not None and crop.size[0] > 2 and crop.size[1] > 2:
                crops.append(crop)
                indices.append(idx)

        if crops:
            batch_results = classify_crops(classifier, crops, imgsz=args.imgsz,
                                           batch_size=args.batch_size)
            for idx, cr in zip(indices, batch_results):
                cls_results[idx] = cr

        cls_time = time.time() - t0
        # Update timing
        for t in image_timings:
            if t["image_id"] == img_id:
                t["cls_time"] = cls_time
                break

    # Fill in any None results
    for i in range(len(cls_results)):
        if cls_results[i] is None:
            cls_results[i] = (0, 0.0, [0], [0.0])

    # ---- Crop-Level Accuracy ----
    print("\n=== Crop-Level Accuracy (detector-matched crops) ===")

    matched = evaluate_crop_accuracy(all_detections, gt_by_img, classifier,
                                     padding=args.padding, imgsz=args.imgsz)

    # For matched pairs, check detector vs classifier accuracy
    det_correct = 0
    cls_correct = 0
    cls_top3_correct = 0
    total_matched = len(matched)

    for mp in matched:
        gt_cls = mp["gt_class"]
        det_cls = mp["det_class"]

        # Find the corresponding classifier result
        # Match by image_id + bbox
        for i, d in enumerate(all_detections):
            if (d["image_id"] == mp["image_id"] and
                d["bbox"] == mp["det_bbox"] and
                d["category_id"] == mp["det_class"]):
                cr = cls_results[i]
                break
        else:
            continue

        if det_cls == gt_cls:
            det_correct += 1
        if cr[0] == gt_cls:
            cls_correct += 1
        if gt_cls in cr[2]:
            cls_top3_correct += 1

    if total_matched > 0:
        print(f"  Matched detections: {total_matched}")
        print(f"  Detector top-1:     {det_correct}/{total_matched} = {det_correct/total_matched:.3f}")
        print(f"  Classifier top-1:   {cls_correct}/{total_matched} = {cls_correct/total_matched:.3f}")
        print(f"  Classifier top-3:   {cls_top3_correct}/{total_matched} = {cls_top3_correct/total_matched:.3f}")

    # ---- Per-Category Analysis ----
    print("\n=== Per-Category: Classifier Fixed vs Broke ===")
    fixed_count = 0
    broke_count = 0

    for mp in matched:
        gt_cls = mp["gt_class"]
        det_cls = mp["det_class"]
        for i, d in enumerate(all_detections):
            if (d["image_id"] == mp["image_id"] and
                d["bbox"] == mp["det_bbox"]):
                cr = cls_results[i]
                break
        else:
            continue

        if det_cls != gt_cls and cr[0] == gt_cls:
            fixed_count += 1
        elif det_cls == gt_cls and cr[0] != gt_cls:
            broke_count += 1

    print(f"  Classifier fixed (det wrong → cls right): {fixed_count}")
    print(f"  Classifier broke (det right → cls wrong):  {broke_count}")
    print(f"  Net benefit: {fixed_count - broke_count}")

    # ---- Fusion Policy Sweep ----
    print("\n=== Fusion Policy Sweep ===")
    print(f"{'Policy':>8} {'cls_t':>6} {'det_t':>6} {'det_mAP':>8} {'cls_mAP':>8} {'hybrid':>8} {'delta':>8}")
    print("-" * 62)

    best_result = None
    best_hybrid = baseline["hybrid_score"]

    cls_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    det_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    for policy in ["A", "B", "C", "D", "E"]:
        if policy in ["D", "E"]:
            threshold_combos = list(itertools.product(cls_thresholds, det_thresholds))
        else:
            threshold_combos = [(ct, 0.7) for ct in cls_thresholds]

        if policy == "A":
            threshold_combos = [(0.0, 0.7)]  # A ignores thresholds

        for cls_t, det_t in threshold_combos:
            fused_preds = []
            for i, det in enumerate(all_detections):
                cr = cls_results[i]
                final_cls, final_score = apply_fusion(
                    det["category_id"], det["score"],
                    cr[0], cr[1],
                    policy=policy, cls_thresh=cls_t, det_thresh=det_t,
                )
                fused_preds.append({
                    "image_id": det["image_id"],
                    "category_id": final_cls,
                    "bbox": det["bbox"],
                    "score": final_score,
                })

            scores = hybrid_score(fused_preds, gt_val)
            delta = scores["hybrid_score"] - baseline["hybrid_score"]

            # Only print promising results
            if delta > -0.01 or policy == "A":
                print(f"{policy:>8} {cls_t:6.2f} {det_t:6.2f} {scores['detection_mAP']:8.4f} "
                      f"{scores['classification_mAP']:8.4f} {scores['hybrid_score']:8.4f} "
                      f"{delta:+8.4f}")

            if scores["hybrid_score"] > best_hybrid:
                best_hybrid = scores["hybrid_score"]
                best_result = {
                    "policy": policy,
                    "cls_threshold": cls_t,
                    "det_threshold": det_t,
                    "det_mAP": scores["detection_mAP"],
                    "cls_mAP": scores["classification_mAP"],
                    "hybrid": scores["hybrid_score"],
                    "delta": delta,
                }

    # ---- Best Result ----
    print(f"\n=== Best Result ===")
    if best_result:
        print(f"  Policy: {best_result['policy']}")
        print(f"  cls_threshold: {best_result['cls_threshold']}")
        print(f"  det_threshold: {best_result['det_threshold']}")
        print(f"  hybrid: {best_result['hybrid']:.4f} (delta: {best_result['delta']:+.4f})")
        print(f"  det_mAP: {best_result['det_mAP']:.4f}")
        print(f"  cls_mAP: {best_result['cls_mAP']:.4f}")
    else:
        print("  No improvement over baseline. Two-stage not justified.")

    # ---- Runtime ----
    print(f"\n=== Runtime ===")
    det_times = [t["det_time"] for t in image_timings]
    cls_times = [t.get("cls_time", 0) for t in image_timings]
    total_times = [d + c for d, c in zip(det_times, cls_times)]

    print(f"  Detection:      avg={np.mean(det_times):.2f}s  p95={np.percentile(det_times, 95):.2f}s  max={max(det_times):.2f}s")
    print(f"  Classification: avg={np.mean(cls_times):.2f}s  p95={np.percentile(cls_times, 95):.2f}s  max={max(cls_times):.2f}s")
    print(f"  Total:          avg={np.mean(total_times):.2f}s  p95={np.percentile(total_times, 95):.2f}s  max={max(total_times):.2f}s")
    print(f"  Extrapolated 50 images: {np.mean(total_times) * 50:.1f}s")
    print(f"  Extrapolated 50 images (p95): {np.percentile(total_times, 95) * 50:.1f}s")

    # ---- Save results ----
    output = {
        "baseline": baseline,
        "best_result": best_result,
        "crop_accuracy": {
            "matched": total_matched,
            "detector_top1": det_correct / total_matched if total_matched else 0,
            "classifier_top1": cls_correct / total_matched if total_matched else 0,
            "classifier_top3": cls_top3_correct / total_matched if total_matched else 0,
            "fixed": fixed_count,
            "broke": broke_count,
        },
        "runtime": {
            "det_avg": float(np.mean(det_times)),
            "cls_avg": float(np.mean(cls_times)),
            "total_avg": float(np.mean(total_times)),
            "total_p95": float(np.percentile(total_times, 95)),
            "total_max": float(max(total_times)),
        },
    }

    out_path = Path(args.classifier).parent.parent / "two_stage_eval.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
