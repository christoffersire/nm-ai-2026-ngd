"""
NorgesGruppen Object Detection — ONNX Entry Point

Uses onnxruntime for inference to avoid PyTorch/ultralytics version issues.

Usage: python run.py --input /data/images --output /output/predictions.json
"""
import argparse
import json
import re
import math
from pathlib import Path

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
}

if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as _f:
        _cfg = json.load(_f)
    _DEFAULTS.update(_cfg)

NC = _DEFAULTS["nc"]
CONF_THRESHOLD = _DEFAULTS["conf_threshold"]
IOU_NMS_THRESHOLD = _DEFAULTS["iou_nms_threshold"]
MAX_PREDICTIONS_PER_IMAGE = _DEFAULTS["max_predictions_per_image"]
MODEL_PATH = SCRIPT_DIR / _DEFAULTS["model_file"]
IMGSZ = 1280


def extract_image_id(filename: str):
    m = re.match(r"img_(\d+)\.(jpg|jpeg|png)", filename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def letterbox(img, new_shape=1280):
    """Resize and pad image to square, maintaining aspect ratio."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2

    resized = np.array(Image.fromarray(img).resize(new_unpad, Image.BILINEAR))

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    padded = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    padded[top:top + new_unpad[1], left:left + new_unpad[0]] = resized

    return padded, r, (left, top)


def nms(boxes, scores, iou_threshold):
    """Non-maximum suppression."""
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


def postprocess(output, r, pad, orig_shape, conf_thresh, iou_thresh, max_det):
    """
    Process YOLOv8 ONNX output.
    Output shape: (1, 4+nc, num_boxes) — transposed from standard.
    """
    # output shape: (1, 360, 33600) for nc=356
    output = output[0]  # remove batch dim
    # Transpose to (num_boxes, 4+nc)
    output = output.T

    # Split into boxes and class scores
    boxes_xywh = output[:, :4]  # cx, cy, w, h
    class_scores = output[:, 4:]  # (num_boxes, nc)

    # Get max class score and class id per box
    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    # Filter by confidence
    mask = max_scores >= conf_thresh
    boxes_xywh = boxes_xywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_xywh) == 0:
        return []

    # Convert from cx,cy,w,h to x,y,w,h
    boxes_xywh_copy = boxes_xywh.copy()
    boxes_xywh_copy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x
    boxes_xywh_copy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y

    # Scale back to original image coordinates
    pad_x, pad_y = pad
    boxes_xywh_copy[:, 0] = (boxes_xywh_copy[:, 0] - pad_x) / r
    boxes_xywh_copy[:, 1] = (boxes_xywh_copy[:, 1] - pad_y) / r
    boxes_xywh_copy[:, 2] = boxes_xywh_copy[:, 2] / r
    boxes_xywh_copy[:, 3] = boxes_xywh_copy[:, 3] / r

    # NMS per class
    final_boxes = []
    final_scores = []
    final_classes = []

    for cls_id in np.unique(class_ids):
        cls_mask = class_ids == cls_id
        cls_boxes = boxes_xywh_copy[cls_mask]
        cls_scores = max_scores[cls_mask]

        keep = nms(cls_boxes, cls_scores, iou_thresh)
        final_boxes.extend(cls_boxes[keep])
        final_scores.extend(cls_scores[keep])
        final_classes.extend([cls_id] * len(keep))

    # Sort by score and limit
    if len(final_scores) > max_det:
        indices = np.argsort(final_scores)[::-1][:max_det]
        final_boxes = [final_boxes[i] for i in indices]
        final_scores = [final_scores[i] for i in indices]
        final_classes = [final_classes[i] for i in indices]

    # Clamp to original image
    oh, ow = orig_shape
    results = []
    for box, score, cls_id in zip(final_boxes, final_scores, final_classes):
        x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        x = max(0.0, x)
        y = max(0.0, y)
        w = max(0.0, min(w, ow - x))
        h = max(0.0, min(h, oh - y))
        if w <= 0 or h <= 0:
            continue
        cls_id = int(cls_id)
        if cls_id < 0 or cls_id >= NC:
            cls_id = 0
        results.append({
            "bbox": [round(x, 1), round(y, 1), round(w, 1), round(h, 1)],
            "score": round(float(score), 4),
            "category_id": cls_id,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load ONNX model
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
    print(f"Model loaded: {MODEL_PATH.name}, input={input_name}")

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
            orig_shape = img_np.shape[:2]

            # Preprocess
            padded, r, pad = letterbox(img_np, IMGSZ)
            blob = padded.astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1)[np.newaxis, ...]  # BCHW

            # Inference
            output = session.run(None, {input_name: blob})

            # Postprocess
            preds = postprocess(output[0], r, pad, orig_shape,
                                CONF_THRESHOLD, IOU_NMS_THRESHOLD,
                                MAX_PREDICTIONS_PER_IMAGE)

            for p in preds:
                p["image_id"] = image_id
                all_predictions.append(p)

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")
            continue

    output_path.write_text(json.dumps(all_predictions))
    print(f"Wrote {len(all_predictions)} predictions for {len(image_files)} images")


if __name__ == "__main__":
    main()
