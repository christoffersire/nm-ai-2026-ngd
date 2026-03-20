"""
NorgesGruppen Object Detection — ONNX TTA Entry Point

Multi-scale TTA with configurable fusion (concat+NMS or WBF).
Falls back to single-pass NMS if runtime is tight.

Usage: python run.py --input /data/images --output /output/predictions.json
"""
import argparse
import json
import re
import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import onnxruntime as ort


# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

_DEFAULTS = {
    "nc": 356,
    "conf_threshold": 0.03,
    "iou_nms_threshold": 0.5,
    "max_predictions_per_image": 500,
    "model_file": "detector.onnx",
    "tta_scales": [1280],
    "tta_flip": False,
    "fusion_method": "concat_nms",
    "wbf_iou_threshold": 0.55,
    "wbf_skip_threshold": 1,
    "time_budget": 260,
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
TTA_SCALES = _DEFAULTS["tta_scales"]
TTA_FLIP = _DEFAULTS["tta_flip"]
FUSION_METHOD = _DEFAULTS["fusion_method"]
WBF_IOU = _DEFAULTS["wbf_iou_threshold"]
WBF_SKIP = _DEFAULTS["wbf_skip_threshold"]
TIME_BUDGET = _DEFAULTS["time_budget"]


def now():
    return datetime.datetime.now().timestamp()


def extract_image_id(filename):
    m = re.match(r"img_(\d+)\.(jpg|jpeg|png)", filename, re.IGNORECASE)
    return int(m.group(1)) if m else None


# --- Preprocessing ---

def letterbox(img, new_shape=1280):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2

    resized = np.array(Image.fromarray(img).resize(new_unpad, Image.BILINEAR))

    top = int(round(dh - 0.1))
    left = int(round(dw - 0.1))

    padded = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    padded[top:top + new_unpad[1], left:left + new_unpad[0]] = resized

    return padded, r, (left, top)


def preprocess(padded):
    blob = padded.astype(np.float32) / 255.0
    return blob.transpose(2, 0, 1)[np.newaxis, ...]


# --- Postprocessing ---

def postprocess_raw(output, r, pad, orig_shape, conf_thresh):
    """Extract raw detections without NMS. Returns list of dicts."""
    output = output[0].T  # (num_boxes, 4+nc)
    boxes_cxcywh = output[:, :4]
    class_scores = output[:, 4:]

    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    mask = max_scores >= conf_thresh
    boxes_cxcywh = boxes_cxcywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_cxcywh) == 0:
        return []

    # cx,cy,w,h -> x,y,w,h
    boxes = boxes_cxcywh.copy()
    boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2

    # Denormalize from model space to original image
    pad_x, pad_y = pad
    boxes[:, 0] = (boxes[:, 0] - pad_x) / r
    boxes[:, 1] = (boxes[:, 1] - pad_y) / r
    boxes[:, 2] = boxes[:, 2] / r
    boxes[:, 3] = boxes[:, 3] / r

    # Clamp to image bounds
    oh, ow = orig_shape
    boxes[:, 0] = np.clip(boxes[:, 0], 0, ow)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, oh)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, ow - boxes[:, 0])
    boxes[:, 3] = np.clip(boxes[:, 3], 0, oh - boxes[:, 1])

    # Filter zero-area
    valid = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
    boxes = boxes[valid]
    max_scores = max_scores[valid]
    class_ids = class_ids[valid]

    preds = []
    for i in range(len(boxes)):
        cat_id = int(class_ids[i])
        if cat_id < 0 or cat_id >= NC:
            cat_id = 0
        preds.append({
            "bbox": [float(boxes[i, 0]), float(boxes[i, 1]),
                     float(boxes[i, 2]), float(boxes[i, 3])],
            "score": float(max_scores[i]),
            "category_id": cat_id,
        })
    return preds


# --- NMS ---

def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def apply_nms(preds, iou_threshold=IOU_NMS_THRESHOLD):
    """Per-class NMS on a list of prediction dicts."""
    if not preds:
        return []

    by_class = defaultdict(list)
    for p in preds:
        by_class[p["category_id"]].append(p)

    results = []
    for cls_id, cls_preds in by_class.items():
        boxes = np.array([p["bbox"] for p in cls_preds])
        scores = np.array([p["score"] for p in cls_preds])
        keep = nms(boxes, scores, iou_threshold)
        results.extend([cls_preds[k] for k in keep])

    return results


# --- WBF ---

def compute_iou_xywh(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0, ix2 - ix) * max(0, iy2 - iy)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def weighted_box_fusion(predictions_list, iou_threshold=0.55, skip_threshold=1):
    """
    WBF across multiple prediction sets (from TTA passes).
    Per-class clustering and weighted-average fusion.
    """
    num_models = len(predictions_list)

    # Tag predictions with model index
    tagged = []
    for model_idx, preds in enumerate(predictions_list):
        for p in preds:
            tagged.append({
                "bbox": p["bbox"],
                "score": p["score"],
                "category_id": p["category_id"],
                "model_idx": model_idx,
            })

    # Group by class
    by_class = defaultdict(list)
    for t in tagged:
        by_class[t["category_id"]].append(t)

    fused = []

    for cls_id, cls_preds in by_class.items():
        cls_preds.sort(key=lambda x: x["score"], reverse=True)

        used = [False] * len(cls_preds)
        clusters = []

        for i in range(len(cls_preds)):
            if used[i]:
                continue

            cluster = [cls_preds[i]]
            used[i] = True

            # Compute current cluster centroid
            def cluster_box(clst):
                weights = np.array([p["score"] for p in clst])
                boxes = np.array([p["bbox"] for p in clst])
                tw = weights.sum()
                if tw == 0:
                    return boxes[0].tolist()
                return (weights[:, None] * boxes).sum(axis=0) / tw

            for j in range(i + 1, len(cls_preds)):
                if used[j]:
                    continue
                cb = cluster_box(cluster)
                iou = compute_iou_xywh(cb, cls_preds[j]["bbox"])
                if iou >= iou_threshold:
                    cluster.append(cls_preds[j])
                    used[j] = True

            # Fuse cluster
            n_models_in = len(set(p["model_idx"] for p in cluster))
            if n_models_in < skip_threshold:
                continue

            weights = np.array([p["score"] for p in cluster])
            boxes = np.array([p["bbox"] for p in cluster])
            tw = weights.sum()
            fused_box = (weights[:, None] * boxes).sum(axis=0) / tw

            avg_score = weights.mean()
            fused_score = float(avg_score * min(n_models_in, num_models) / num_models)

            fused.append({
                "bbox": fused_box.tolist(),
                "score": fused_score,
                "category_id": cls_id,
            })

    return fused


# --- TTA Inference ---

def run_single_pass(session, input_name, img_np, orig_shape, scale, flip=False):
    """Run inference at one scale/flip. Returns list of prediction dicts."""
    if flip:
        img_np = img_np[:, ::-1, :].copy()

    padded, r, pad = letterbox(img_np, scale)
    blob = preprocess(padded)
    output = session.run(None, {input_name: blob})
    preds = postprocess_raw(output[0], r, pad, orig_shape, CONF_THRESHOLD)

    if flip:
        ow = orig_shape[1]
        for p in preds:
            x, y, w, h = p["bbox"]
            p["bbox"] = [ow - x - w, y, w, h]

    return preds


def run_tta(session, input_name, img_np, orig_shape, scales, flip):
    """Run multi-scale TTA. Returns list of lists (one per pass)."""
    all_passes = []
    for scale in scales:
        preds = run_single_pass(session, input_name, img_np, orig_shape, scale)
        all_passes.append(preds)

    if flip:
        # Flip at the native/middle scale
        flip_scale = scales[len(scales) // 2] if len(scales) > 1 else scales[0]
        preds = run_single_pass(session, input_name, img_np, orig_shape, flip_scale, flip=True)
        all_passes.append(preds)

    return all_passes


def fuse_predictions(all_passes, method="concat_nms"):
    """Fuse predictions from multiple TTA passes."""
    if method == "concat_nms":
        # Concatenate all passes, apply per-class NMS
        concat = []
        for preds in all_passes:
            concat.extend(preds)
        return apply_nms(concat)

    elif method == "wbf":
        # Raw WBF across passes
        return weighted_box_fusion(all_passes, iou_threshold=WBF_IOU,
                                   skip_threshold=WBF_SKIP)

    elif method == "nms_then_wbf":
        # Per-pass NMS first, then WBF across passes
        nmsed = [apply_nms(preds) for preds in all_passes]
        return weighted_box_fusion(nmsed, iou_threshold=WBF_IOU,
                                   skip_threshold=WBF_SKIP)

    else:
        # Fallback: single-pass NMS
        return apply_nms(all_passes[0]) if all_passes else []


# --- Main ---

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

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(str(MODEL_PATH), providers=providers)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        output_path.write_text("[]")
        return

    input_name = session.get_inputs()[0].name

    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    n_images = len(image_files)
    print(f"Model: {MODEL_PATH.name}, images: {n_images}")
    print(f"TTA scales: {TTA_SCALES}, flip: {TTA_FLIP}, fusion: {FUSION_METHOD}")

    # --- Runtime calibration ---
    start_time = now()
    calibration_times = []

    # Determine initial mode
    active_scales = list(TTA_SCALES)
    active_flip = TTA_FLIP
    active_fusion = FUSION_METHOD

    all_predictions = []

    for idx, img_path in enumerate(image_files):
        image_id = extract_image_id(img_path.name)
        if image_id is None:
            continue

        t0 = now()

        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            orig_shape = img_np.shape[:2]

            # Run TTA
            all_passes = run_tta(session, input_name, img_np, orig_shape,
                                 active_scales, active_flip)

            # Fuse
            if len(all_passes) == 1 and active_fusion != "concat_nms":
                # Single pass: use proven NMS, not WBF
                preds = apply_nms(all_passes[0])
            else:
                preds = fuse_predictions(all_passes, active_fusion)

            # Apply prediction cap
            if len(preds) > MAX_PREDS:
                preds.sort(key=lambda x: x["score"], reverse=True)
                preds = preds[:MAX_PREDS]

            # Round values
            for p in preds:
                p["image_id"] = image_id
                p["bbox"] = [round(v, 1) for v in p["bbox"]]
                p["score"] = round(p["score"], 4)

            all_predictions.extend(preds)

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")
            continue

        elapsed_img = now() - t0
        calibration_times.append(elapsed_img)

        # Runtime check after calibration batch (first 5 images)
        if idx == min(4, n_images - 1) and n_images > 5:
            p75_time = sorted(calibration_times)[int(len(calibration_times) * 0.75)]
            remaining = n_images - idx - 1
            projected = (now() - start_time) + p75_time * remaining * 1.1  # 10% safety

            if projected > TIME_BUDGET:
                # Downgrade: try dropping to fewer scales
                if len(active_scales) > 1:
                    # Keep only the middle scale + flip
                    mid = active_scales[len(active_scales) // 2]
                    active_scales = [mid]
                    active_flip = TTA_FLIP
                    print(f"[RUNTIME] Downgrading to single-scale + flip (projected {projected:.0f}s > {TIME_BUDGET}s)")
                else:
                    active_flip = False
                    active_fusion = "concat_nms"
                    print(f"[RUNTIME] Downgrading to single-pass NMS (projected {projected:.0f}s > {TIME_BUDGET}s)")

    output_path.write_text(json.dumps(all_predictions))
    total_time = now() - start_time
    print(f"Wrote {len(all_predictions)} predictions for {n_images} images in {total_time:.1f}s")


if __name__ == "__main__":
    main()
