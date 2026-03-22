"""
WBF parameter sweep on cached model predictions.

Step 1: Run both models on val images with very low conf, cache raw outputs
Step 2: Sweep WBF params (conf, iou, weights, max_preds) on cached outputs
Step 3: Score each config with local scorer

Usage:
  python scoring/wbf_sweep.py
"""
import json
import time
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import onnxruntime as ort
from ensemble_boxes import weighted_boxes_fusion

# Add project root to path for imports
import sys
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scoring"))
from scorer import hybrid_score

RAW_DIR = PROJECT_DIR / "data" / "raw"
ANNOTATIONS_PATH = RAW_DIR / "annotations_fixed.json"
VAL_SPLIT_PATH = PROJECT_DIR / "data" / "val_split.json"
IMAGE_DIR = RAW_DIR / "images"
MODEL_A = PROJECT_DIR / "weights" / "v3-1280.onnx"
MODEL_B = PROJECT_DIR / "weights" / "v3-1536-fp16.onnx"
IMGSZ_A = 1280
IMGSZ_B = 1536


def letterbox(img_np, new_shape):
    h, w = img_np.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2
    resized = np.array(Image.fromarray(img_np).resize(new_unpad, Image.BILINEAR))
    top = int(round(dh - 0.1))
    left = int(round(dw - 0.1))
    padded = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    padded[top:top + new_unpad[1], left:left + new_unpad[0]] = resized
    return padded, r, (left, top)


def run_model(session, input_name, img_np, imgsz, min_conf=0.001):
    """Run model and return raw detections (very low conf to cache everything)."""
    oh, ow = img_np.shape[:2]
    padded, r, (pad_x, pad_y) = letterbox(img_np, imgsz)
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

    output = session.run(None, {input_name: blob})
    output = output[0][0].T

    boxes_cxcywh = output[:, :4]
    class_scores = output[:, 4:]
    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    mask = max_scores >= min_conf
    boxes_cxcywh = boxes_cxcywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_cxcywh) == 0:
        return np.array([]).reshape(0, 4), np.array([]), np.array([])

    # Convert to normalized [x1,y1,x2,y2] for WBF
    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1 = (cx - w / 2 - pad_x) / r
    y1 = (cy - h / 2 - pad_y) / r
    x2 = (cx + w / 2 - pad_x) / r
    y2 = (cy + h / 2 - pad_y) / r

    boxes_norm = np.stack([
        np.clip(x1 / ow, 0, 1),
        np.clip(y1 / oh, 0, 1),
        np.clip(x2 / ow, 0, 1),
        np.clip(y2 / oh, 0, 1),
    ], axis=1)

    valid = (boxes_norm[:, 2] > boxes_norm[:, 0]) & (boxes_norm[:, 3] > boxes_norm[:, 1])
    return boxes_norm[valid], max_scores[valid], class_ids[valid]


def run_with_flip(session, input_name, img_np, imgsz, min_conf=0.001):
    """Run model + horizontal flip TTA."""
    boxes, scores, labels = run_model(session, input_name, img_np, imgsz, min_conf)

    img_flip = img_np[:, ::-1, :].copy()
    boxes_f, scores_f, labels_f = run_model(session, input_name, img_flip, imgsz, min_conf)

    if len(boxes_f) > 0:
        # Flip x coordinates back
        boxes_f_fixed = boxes_f.copy()
        boxes_f_fixed[:, 0] = 1 - boxes_f[:, 2]
        boxes_f_fixed[:, 2] = 1 - boxes_f[:, 0]
        boxes_f = boxes_f_fixed

    if len(boxes) == 0:
        return boxes_f, scores_f, labels_f
    if len(boxes_f) == 0:
        return boxes, scores, labels

    return (
        np.concatenate([boxes, boxes_f]),
        np.concatenate([scores, scores_f]),
        np.concatenate([labels, labels_f]),
    )


def main():
    # Load data
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)
    with open(VAL_SPLIT_PATH) as f:
        val_split = json.load(f)

    val_ids = set(val_split["val_ids"])
    id_to_img = {img["id"]: img for img in data["images"]}
    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}

    # Build GT for val images
    gt_by_img = defaultdict(list)
    for ann in data["annotations"]:
        if ann["image_id"] in val_ids:
            gt_by_img[ann["image_id"]].append(ann)

    val_images = sorted([img for img in data["images"] if img["id"] in val_ids], key=lambda x: x["id"])
    print(f"Val images: {len(val_images)}")

    # Load models
    print("Loading models...")
    providers = ["CPUExecutionProvider"]
    session_a = ort.InferenceSession(str(MODEL_A), providers=providers)
    input_a = session_a.get_inputs()[0].name
    session_b = ort.InferenceSession(str(MODEL_B), providers=providers)
    input_b = session_b.get_inputs()[0].name
    print("Models loaded")

    # Step 1: Cache raw predictions for all val images
    print("\nCaching predictions (this takes a few minutes on CPU)...")
    cache = {}
    t0 = time.time()
    for idx, img_info in enumerate(val_images):
        img_path = IMAGE_DIR / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        oh, ow = img_np.shape[:2]

        # Model A with TTA flip (matching v13 config)
        boxes_a, scores_a, labels_a = run_with_flip(session_a, input_a, img_np, IMGSZ_A)
        # Model B without TTA (matching v13 config)
        boxes_b, scores_b, labels_b = run_model(session_b, input_b, img_np, IMGSZ_B)

        cache[img_info["id"]] = {
            "oh": oh, "ow": ow,
            "a_boxes": boxes_a, "a_scores": scores_a, "a_labels": labels_a,
            "b_boxes": boxes_b, "b_scores": scores_b, "b_labels": labels_b,
        }

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(val_images) - idx - 1) / rate
            print(f"  [{idx+1}/{len(val_images)}] {rate:.1f} img/s, ETA: {eta:.0f}s")

    print(f"Cached {len(cache)} images in {time.time()-t0:.0f}s")

    # Step 2: Sweep WBF params
    conf_values = [0.01, 0.03, 0.05, 0.08]
    iou_values = [0.50, 0.55, 0.60]
    weight_values = [[1.0, 1.0], [1.0, 1.2], [1.0, 1.5]]
    max_pred_values = [300, 500]

    # Build GT in scorer format
    gt_list = []
    for img_info in val_images:
        img_preds = []
        for ann in gt_by_img[img_info["id"]]:
            img_preds.append({
                "image_id": img_info["id"],
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
            })
        gt_list.extend(img_preds)

    results = []
    combos = list(itertools.product(conf_values, iou_values, weight_values, max_pred_values))
    print(f"\nSweeping {len(combos)} WBF configs...")

    for ci, (conf, wbf_iou, weights, max_preds) in enumerate(combos):
        all_preds = []
        for img_info in val_images:
            c = cache[img_info["id"]]
            oh, ow = c["oh"], c["ow"]

            # Filter by conf
            mask_a = c["a_scores"] >= conf
            mask_b = c["b_scores"] >= conf

            a_boxes = c["a_boxes"][mask_a]
            a_scores = c["a_scores"][mask_a]
            a_labels = c["a_labels"][mask_a]
            b_boxes = c["b_boxes"][mask_b]
            b_scores = c["b_scores"][mask_b]
            b_labels = c["b_labels"][mask_b]

            all_boxes = [a_boxes.tolist() if len(a_boxes) > 0 else [],
                         b_boxes.tolist() if len(b_boxes) > 0 else []]
            all_scores_list = [a_scores.tolist() if len(a_scores) > 0 else [],
                               b_scores.tolist() if len(b_scores) > 0 else []]
            all_labels_list = [a_labels.astype(int).tolist() if len(a_labels) > 0 else [],
                               b_labels.astype(int).tolist() if len(b_labels) > 0 else []]

            if any(len(b) > 0 for b in all_boxes):
                boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
                    all_boxes, all_scores_list, all_labels_list,
                    weights=weights, iou_thr=wbf_iou, skip_box_thr=conf,
                )
                for j in range(min(len(boxes_fused), max_preds)):
                    x1 = float(boxes_fused[j][0] * ow)
                    y1 = float(boxes_fused[j][1] * oh)
                    x2 = float(boxes_fused[j][2] * ow)
                    y2 = float(boxes_fused[j][3] * oh)
                    w = x2 - x1
                    h = y2 - y1
                    if w <= 0 or h <= 0:
                        continue
                    all_preds.append({
                        "image_id": img_info["id"],
                        "category_id": int(labels_fused[j]),
                        "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                        "score": round(float(scores_fused[j]), 4),
                    })

        # Score
        try:
            score = hybrid_score(all_preds, gt_list)
            det_map = score.get("detection_mAP", 0) if isinstance(score, dict) else 0
            cls_map = score.get("classification_mAP", 0) if isinstance(score, dict) else 0
            hybrid = score.get("hybrid_score", 0) if isinstance(score, dict) else score
        except Exception as e:
            hybrid = 0
            det_map = 0
            cls_map = 0

        results.append({
            "conf": conf, "wbf_iou": wbf_iou, "weights": weights,
            "max_preds": max_preds, "hybrid": round(hybrid, 6),
            "det_mAP": round(det_map, 6), "cls_mAP": round(cls_map, 6),
            "n_preds": len(all_preds),
        })

        if (ci + 1) % 10 == 0:
            print(f"  [{ci+1}/{len(combos)}] configs tested")

    # Sort and display
    results.sort(key=lambda x: -x["hybrid"])

    print(f"\n{'='*80}")
    print(f"TOP 10 CONFIGS (out of {len(results)}):")
    print(f"{'Rank':>4} {'Hybrid':>8} {'Det':>7} {'Cls':>7} {'Conf':>5} {'WBF':>5} {'Weights':>12} {'Max':>5} {'#Pred':>6}")
    print("-" * 80)
    for i, r in enumerate(results[:10]):
        w_str = f"[{r['weights'][0]},{r['weights'][1]}]"
        print(f"{i+1:>4} {r['hybrid']:>8.5f} {r['det_mAP']:>7.4f} {r['cls_mAP']:>7.4f} "
              f"{r['conf']:>5.2f} {r['wbf_iou']:>5.2f} {w_str:>12} {r['max_preds']:>5} {r['n_preds']:>6}")

    # Show v13 baseline for comparison
    v13_config = {"conf": 0.05, "wbf_iou": 0.55, "weights": [1.0, 1.2], "max_preds": 500}
    v13_result = [r for r in results if r["conf"] == 0.05 and r["wbf_iou"] == 0.55
                  and r["weights"] == [1.0, 1.2] and r["max_preds"] == 500]
    if v13_result:
        print(f"\nV13 baseline: hybrid={v13_result[0]['hybrid']:.5f}")

    # Save
    with open(PROJECT_DIR / "scoring" / "wbf_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to scoring/wbf_sweep_results.json")


if __name__ == "__main__":
    main()
