"""
NorgesGruppen Object Detection — Tiled ONNX Inference

Tiles each input image, runs ONNX inference on each tile,
maps detections back to full-image coordinates, and fuses
with NMS or WBF.

Uses shared tile geometry from data/tiling.py to ensure
train/infer consistency.

Usage: python run.py --input /data/images --output /output/predictions.json
"""
import argparse
import json
import re
import math
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
    "conf_threshold": 0.05,
    "iou_nms_threshold": 0.5,
    "max_predictions_per_image": 500,
    "model_file": "detector.onnx",
    "tile_size": 1536,
    "tile_stride": 768,
    "fusion_method": "nms",
    "wbf_iou_threshold": 0.55,
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
TILE_SIZE = _DEFAULTS["tile_size"]
TILE_STRIDE = _DEFAULTS["tile_stride"]
FUSION_METHOD = _DEFAULTS["fusion_method"]
WBF_IOU = _DEFAULTS["wbf_iou_threshold"]

# Inference imgsz — tiles are resized to this for ONNX
INFER_IMGSZ = 1280


def extract_image_id(filename):
    m = re.match(r"img_(\d+)\.(jpg|jpeg|png)", filename, re.IGNORECASE)
    return int(m.group(1)) if m else None


# --- Shared tile geometry (must match prepare_tiles.py) ---

def generate_tile_coords(img_w, img_h, tile_size, stride):
    """Same function as data/tiling.py — duplicated to avoid import path issues in sandbox."""
    tiles = set()
    for y in range(0, img_h - tile_size + 1, stride):
        for x in range(0, img_w - tile_size + 1, stride):
            tiles.add((x, y))
    if img_w > tile_size:
        right_x = img_w - tile_size
        for y in range(0, img_h - tile_size + 1, stride):
            tiles.add((right_x, y))
    if img_h > tile_size:
        bottom_y = img_h - tile_size
        for x in range(0, img_w - tile_size + 1, stride):
            tiles.add((x, bottom_y))
    if img_w > tile_size and img_h > tile_size:
        tiles.add((img_w - tile_size, img_h - tile_size))
    if img_w <= tile_size or img_h <= tile_size:
        tiles.add((0, 0))
    return sorted(tiles)


# --- ONNX inference ---

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


def run_tile_inference(session, input_name, tile_np, conf_thresh):
    """Run ONNX on a single tile crop. Returns list of [x, y, w, h, score, class_id] in tile coordinates."""
    tile_h, tile_w = tile_np.shape[:2]

    padded, r, pad = letterbox(tile_np, INFER_IMGSZ)
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

    output = session.run(None, {input_name: blob})
    output = output[0][0].T  # (num_boxes, 4+nc)

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

    # Convert from model space to tile pixel coordinates
    pad_x, pad_y = pad
    detections = []
    for i in range(len(boxes_cxcywh)):
        cx, cy, w, h = boxes_cxcywh[i]
        x1 = (cx - w / 2 - pad_x) / r
        y1 = (cy - h / 2 - pad_y) / r
        det_w = w / r
        det_h = h / r

        # Clamp to tile bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        det_w = min(det_w, tile_w - x1)
        det_h = min(det_h, tile_h - y1)

        if det_w <= 0 or det_h <= 0:
            continue

        cat_id = int(class_ids[i])
        if cat_id < 0 or cat_id >= NC:
            cat_id = 0

        detections.append([float(x1), float(y1), float(det_w), float(det_h),
                           float(max_scores[i]), cat_id])

    return detections


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


def fuse_detections_nms(all_dets, img_w, img_h, iou_thresh=0.5):
    """Fuse detections from all tiles using per-class NMS."""
    if not all_dets:
        return []

    by_class = defaultdict(list)
    for det in all_dets:
        by_class[det[5]].append(det)

    results = []
    for cls_id, dets in by_class.items():
        boxes = np.array([[d[0], d[1], d[2], d[3]] for d in dets])
        scores = np.array([d[4] for d in dets])
        keep = nms(boxes, scores, iou_thresh)
        for k in keep:
            d = dets[k]
            results.append(d)

    return results


def fuse_detections_wbf(all_dets, img_w, img_h, iou_thresh=0.55):
    """Fuse detections from all tiles using WBF."""
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        return fuse_detections_nms(all_dets, img_w, img_h, iou_thresh)

    if not all_dets:
        return []

    # WBF expects [x1,y1,x2,y2] normalized to [0,1]
    boxes_norm = []
    scores = []
    labels = []

    for det in all_dets:
        x, y, w, h, score, cls_id = det
        x1 = x / img_w
        y1 = y / img_h
        x2 = (x + w) / img_w
        y2 = (y + h) / img_h
        boxes_norm.append([
            max(0, min(1, x1)), max(0, min(1, y1)),
            max(0, min(1, x2)), max(0, min(1, y2))
        ])
        scores.append(score)
        labels.append(cls_id)

    # WBF treats input as list of lists (one per "model")
    # For tile fusion, all tiles are one "model"
    bf, sf, lf = weighted_boxes_fusion(
        [boxes_norm], [scores], [labels],
        iou_thr=iou_thresh, skip_box_thr=CONF_THRESHOLD
    )

    results = []
    for i in range(len(bf)):
        x1 = float(bf[i][0] * img_w)
        y1 = float(bf[i][1] * img_h)
        x2 = float(bf[i][2] * img_w)
        y2 = float(bf[i][3] * img_h)
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        results.append([x1, y1, w, h, float(sf[i]), int(lf[i])])

    return results


# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    print(f"Model: {MODEL_PATH.name}, tile={TILE_SIZE}, stride={TILE_STRIDE}, fusion={FUSION_METHOD}")

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
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            img_h, img_w = img_np.shape[:2]

            # Generate tiles (same geometry as training)
            tile_coords = generate_tile_coords(img_w, img_h, TILE_SIZE, TILE_STRIDE)

            # Run inference on each tile
            all_tile_dets = []
            for tx, ty in tile_coords:
                # Crop tile
                tile_crop = img_np[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE]
                if tile_crop.shape[0] == 0 or tile_crop.shape[1] == 0:
                    continue

                # Run inference
                tile_dets = run_tile_inference(session, input_name, tile_crop, CONF_THRESHOLD)

                # Map to full-image coordinates
                for det in tile_dets:
                    det[0] += tx  # x
                    det[1] += ty  # y
                    all_tile_dets.append(det)

            # Fuse detections from all tiles
            if FUSION_METHOD == "wbf":
                fused = fuse_detections_wbf(all_tile_dets, img_w, img_h, WBF_IOU)
            else:
                fused = fuse_detections_nms(all_tile_dets, img_w, img_h, IOU_NMS_THRESHOLD)

            # Sort by score, cap at max predictions
            fused.sort(key=lambda d: d[4], reverse=True)
            fused = fused[:MAX_PREDS]

            # Format output
            for det in fused:
                x, y, w, h, score, cls_id = det
                all_predictions.append({
                    "image_id": image_id,
                    "category_id": int(cls_id),
                    "bbox": [round(x, 1), round(y, 1), round(w, 1), round(h, 1)],
                    "score": round(score, 4),
                })

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")
            continue

    output_path.write_text(json.dumps(all_predictions))
    print(f"Wrote {len(all_predictions)} predictions for {len(image_files)} images")


if __name__ == "__main__":
    main()
