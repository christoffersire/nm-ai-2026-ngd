"""
NorgesGruppen Object Detection — ONNX + ensemble_boxes WBF

Uses onnxruntime for inference with ensemble_boxes WBF for postprocessing.
ensemble_boxes 1.0.9 is pre-installed in the sandbox.

Usage: python run.py --input /data/images --output /output/predictions.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort
from ensemble_boxes import weighted_boxes_fusion


# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

_DEFAULTS = {
    "nc": 356,
    "conf_threshold": 0.05,
    "iou_nms_threshold": 0.5,
    "max_predictions_per_image": 500,
    "model_file": "detector.onnx",
    "wbf_iou_threshold": 0.5,
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
WBF_IOU = _DEFAULTS["wbf_iou_threshold"]
IMGSZ = 1280


def extract_image_id(filename):
    m = re.match(r"img_(\d+)\.(jpg|jpeg|png)", filename, re.IGNORECASE)
    return int(m.group(1)) if m else None


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


def postprocess_to_normalized(output, r, pad, orig_shape, conf_thresh):
    """
    Process ONNX output into normalized [0,1] boxes for ensemble_boxes.
    Returns (boxes_norm, scores, labels) where boxes are [x1,y1,x2,y2] in [0,1].
    """
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
        return np.array([]).reshape(0, 4), np.array([]), np.array([])

    # cx,cy,w,h -> x1,y1,x2,y2
    boxes = np.zeros_like(boxes_cxcywh)
    boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2  # x1
    boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2  # y1
    boxes[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2  # x2
    boxes[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2  # y2

    # Denormalize from model space to original image
    pad_x, pad_y = pad
    oh, ow = orig_shape
    boxes[:, 0] = (boxes[:, 0] - pad_x) / r
    boxes[:, 1] = (boxes[:, 1] - pad_y) / r
    boxes[:, 2] = (boxes[:, 2] - pad_x) / r
    boxes[:, 3] = (boxes[:, 3] - pad_y) / r

    # Clamp to image bounds
    boxes[:, 0] = np.clip(boxes[:, 0], 0, ow)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, oh)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, ow)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, oh)

    # Filter zero-area
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    max_scores = max_scores[valid]
    class_ids = class_ids[valid]

    # Normalize to [0, 1] for ensemble_boxes
    boxes_norm = boxes.copy()
    boxes_norm[:, 0] /= ow
    boxes_norm[:, 1] /= oh
    boxes_norm[:, 2] /= ow
    boxes_norm[:, 3] /= oh

    # Clip to [0, 1]
    boxes_norm = np.clip(boxes_norm, 0, 1)

    return boxes_norm, max_scores, class_ids


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
    print(f"Model: {MODEL_PATH.name}, WBF IoU: {WBF_IOU}")

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
            orig_shape = img_np.shape[:2]  # (h, w)
            oh, ow = orig_shape

            # Preprocess
            padded, r, pad = letterbox(img_np, IMGSZ)
            blob = padded.astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

            # Inference
            output = session.run(None, {input_name: blob})

            # Get normalized boxes for WBF
            boxes_norm, scores, labels = postprocess_to_normalized(
                output[0], r, pad, orig_shape, CONF_THRESHOLD
            )

            if len(boxes_norm) == 0:
                continue

            # WBF with single model (acts as smart NMS)
            boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
                [boxes_norm.tolist()],
                [scores.tolist()],
                [labels.astype(int).tolist()],
                iou_thr=WBF_IOU,
                skip_box_thr=CONF_THRESHOLD,
            )

            # Convert back to COCO format [x, y, w, h]
            for i in range(min(len(boxes_fused), MAX_PREDS)):
                x1 = float(boxes_fused[i][0] * ow)
                y1 = float(boxes_fused[i][1] * oh)
                x2 = float(boxes_fused[i][2] * ow)
                y2 = float(boxes_fused[i][3] * oh)
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue
                cat_id = int(labels_fused[i])
                if cat_id < 0 or cat_id >= NC:
                    cat_id = 0
                all_predictions.append({
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                    "score": round(float(scores_fused[i]), 4),
                })

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")
            continue

    output_path.write_text(json.dumps(all_predictions))
    print(f"Wrote {len(all_predictions)} predictions for {len(image_files)} images")


if __name__ == "__main__":
    main()
