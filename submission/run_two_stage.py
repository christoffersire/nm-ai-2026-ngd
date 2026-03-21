"""
NorgesGruppen Object Detection — Two-Stage ONNX Pipeline

Stage 1: YOLO detector (finds bounding boxes with class predictions)
Stage 2: Classifier (re-labels each crop for better classification accuracy)

Fusion: classifier overrides detector class when confident, keeps detector
class otherwise. Detection score is always from the detector.

Usage: python run.py --input /data/images --output /output/predictions.json
"""
import argparse
import json
import re
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
    "detector_file": "detector.onnx",
    "classifier_file": "classifier.onnx",
    "det_imgsz": 1280,
    "cls_imgsz": 224,
    "cls_conf_threshold": 0.8,
    "cls_override_always": False,
    "crop_padding": 0.1,
    "tta_flip": True,
}

if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as _f:
        _cfg = json.load(_f)
    _DEFAULTS.update(_cfg)

NC = _DEFAULTS["nc"]
CONF_THRESHOLD = _DEFAULTS["conf_threshold"]
IOU_NMS_THRESHOLD = _DEFAULTS["iou_nms_threshold"]
MAX_PREDS = _DEFAULTS["max_predictions_per_image"]
DETECTOR_PATH = SCRIPT_DIR / _DEFAULTS["detector_file"]
CLASSIFIER_PATH = SCRIPT_DIR / _DEFAULTS["classifier_file"]
DET_IMGSZ = _DEFAULTS["det_imgsz"]
CLS_IMGSZ = _DEFAULTS["cls_imgsz"]
CLS_CONF_THRESHOLD = _DEFAULTS["cls_conf_threshold"]
CLS_OVERRIDE_ALWAYS = _DEFAULTS["cls_override_always"]
CROP_PADDING = _DEFAULTS["crop_padding"]
TTA_FLIP = _DEFAULTS["tta_flip"]


def extract_image_id(filename):
    m = re.match(r"img_(\d+)\.(jpg|jpeg|png)", filename, re.IGNORECASE)
    return int(m.group(1)) if m else None


# --- Detection ---

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


def detect(session, input_name, img_np, orig_shape, imgsz, conf_thresh, iou_thresh, flip=False):
    """Run YOLO detector. Returns list of prediction dicts."""
    if flip:
        img_np = img_np[:, ::-1, :].copy()

    oh, ow = orig_shape
    padded, r, pad = letterbox(img_np, imgsz)
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

    # cx,cy,w,h -> x,y,w,h in original image coords
    pad_x, pad_y = pad
    boxes = boxes_cxcywh.copy()
    boxes[:, 0] = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2 - pad_x) / r
    boxes[:, 1] = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2 - pad_y) / r
    boxes[:, 2] = boxes_cxcywh[:, 2] / r
    boxes[:, 3] = boxes_cxcywh[:, 3] / r

    if flip:
        boxes[:, 0] = ow - boxes[:, 0] - boxes[:, 2]

    # Clamp
    boxes[:, 0] = np.clip(boxes[:, 0], 0, ow)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, oh)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, ow - boxes[:, 0])
    boxes[:, 3] = np.clip(boxes[:, 3], 0, oh - boxes[:, 1])

    valid = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
    boxes = boxes[valid]
    max_scores = max_scores[valid]
    class_ids = class_ids[valid]

    # Per-class NMS
    final = []
    for cls_id in np.unique(class_ids):
        cls_mask = class_ids == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = max_scores[cls_mask]
        keep = nms(cls_boxes, cls_scores, iou_thresh)
        for k in keep:
            cat = int(cls_id)
            if cat < 0 or cat >= NC:
                cat = 0
            final.append({
                "bbox": [float(cls_boxes[k, 0]), float(cls_boxes[k, 1]),
                         float(cls_boxes[k, 2]), float(cls_boxes[k, 3])],
                "score": float(cls_scores[k]),
                "category_id": cat,
            })

    return final


def merge_flip_preds(preds_orig, preds_flip, iou_thresh=0.5):
    """Merge original and flipped predictions: average boxes where IoU > thresh."""
    if not preds_flip:
        return preds_orig

    # Simple approach: concat + per-class NMS (keeps the higher-scoring box)
    all_preds = preds_orig + preds_flip
    by_class = {}
    for p in all_preds:
        by_class.setdefault(p["category_id"], []).append(p)

    merged = []
    for cls_id, cls_preds in by_class.items():
        boxes = np.array([p["bbox"] for p in cls_preds])
        scores = np.array([p["score"] for p in cls_preds])
        keep = nms(boxes, scores, iou_thresh)
        merged.extend([cls_preds[k] for k in keep])

    return merged


# --- Classification ---

def classify_crops(cls_session, cls_input_name, img, preds, imgsz=224, padding=0.1):
    """Classify each detection crop. Returns (class_id, confidence) per prediction."""
    results = []

    for p in preds:
        x, y, w, h = p["bbox"]
        img_w, img_h = img.size

        # Crop with padding
        pad_x = w * padding
        pad_y = h * padding
        x1 = max(0, int(x - pad_x))
        y1 = max(0, int(y - pad_y))
        x2 = min(img_w, int(x + w + pad_x))
        y2 = min(img_h, int(y + h + pad_y))

        if x2 <= x1 or y2 <= y1:
            results.append((p["category_id"], 0.0))
            continue

        crop = img.crop((x1, y1, x2, y2))
        crop = crop.resize((imgsz, imgsz), Image.BILINEAR)

        # Preprocess: HWC -> CHW, normalize, add batch dim
        blob = np.array(crop, dtype=np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

        # Run classifier
        try:
            output = cls_session.run(None, {cls_input_name: blob})
            probs = output[0][0]  # (num_classes,)

            # Softmax if not already
            if probs.min() < 0 or probs.sum() > 1.5:
                probs = np.exp(probs - probs.max())
                probs = probs / probs.sum()

            cls_id = int(probs.argmax())
            cls_conf = float(probs[cls_id])
            results.append((cls_id, cls_conf))
        except Exception:
            results.append((p["category_id"], 0.0))

    return results


def fuse_predictions(preds, cls_results, cls_conf_thresh, override_always=False):
    """Fuse detector and classifier predictions."""
    for i, (cls_id, cls_conf) in enumerate(cls_results):
        if override_always or cls_conf >= cls_conf_thresh:
            if 0 <= cls_id < NC:
                preds[i]["category_id"] = cls_id
        # Always keep detector score for ranking
    return preds


# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Load detector
    if not DETECTOR_PATH.exists():
        print(f"[ERROR] Detector not found: {DETECTOR_PATH}")
        output_path.write_text("[]")
        return

    det_session = ort.InferenceSession(str(DETECTOR_PATH), providers=providers)
    det_input = det_session.get_inputs()[0].name
    print(f"Detector: {DETECTOR_PATH.name}, imgsz={DET_IMGSZ}")

    # Load classifier (optional)
    cls_session = None
    cls_input = None
    if CLASSIFIER_PATH.exists():
        try:
            cls_session = ort.InferenceSession(str(CLASSIFIER_PATH), providers=providers)
            cls_input = cls_session.get_inputs()[0].name
            print(f"Classifier: {CLASSIFIER_PATH.name}, imgsz={CLS_IMGSZ}, threshold={CLS_CONF_THRESHOLD}")
        except Exception as e:
            print(f"[WARN] Failed to load classifier: {e}")
    else:
        print("[INFO] No classifier — running detector-only mode")

    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    print(f"Processing {len(image_files)} images, TTA flip={TTA_FLIP}")
    all_predictions = []

    for img_path in image_files:
        image_id = extract_image_id(img_path.name)
        if image_id is None:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            orig_shape = img_np.shape[:2]

            # Stage 1: Detection
            preds = detect(det_session, det_input, img_np, orig_shape,
                           DET_IMGSZ, CONF_THRESHOLD, IOU_NMS_THRESHOLD)

            # TTA: horizontal flip
            if TTA_FLIP:
                preds_flip = detect(det_session, det_input, img_np, orig_shape,
                                    DET_IMGSZ, CONF_THRESHOLD, IOU_NMS_THRESHOLD, flip=True)
                preds = merge_flip_preds(preds, preds_flip, IOU_NMS_THRESHOLD)

            # Stage 2: Classification (re-label crops)
            if cls_session is not None and preds:
                cls_results = classify_crops(cls_session, cls_input, img, preds,
                                             CLS_IMGSZ, CROP_PADDING)
                preds = fuse_predictions(preds, cls_results,
                                         CLS_CONF_THRESHOLD, CLS_OVERRIDE_ALWAYS)

            # Cap predictions
            if len(preds) > MAX_PREDS:
                preds.sort(key=lambda x: x["score"], reverse=True)
                preds = preds[:MAX_PREDS]

            # Finalize
            for p in preds:
                p["image_id"] = image_id
                p["bbox"] = [round(v, 1) for v in p["bbox"]]
                p["score"] = round(p["score"], 4)

            all_predictions.extend(preds)

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")
            continue

    output_path.write_text(json.dumps(all_predictions))
    print(f"Wrote {len(all_predictions)} predictions for {len(image_files)} images")


if __name__ == "__main__":
    main()
