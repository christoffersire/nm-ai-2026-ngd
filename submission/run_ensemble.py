"""
NorgesGruppen Object Detection — Multi-Model ONNX Ensemble with WBF

Supports 2 or 3 ONNX detector models. Backward-compatible with the
existing model_a/model_b config format used by the current best setup.

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


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

_DEFAULTS = {
    "nc": 356,
    "conf_threshold": 0.05,
    "max_predictions_per_image": 500,
    "model_a": "model_a.onnx",
    "model_b": "model_b.onnx",
    "imgsz_a": 1280,
    "imgsz_b": 1280,
    "wbf_iou_threshold": 0.55,
    "wbf_skip_threshold": 0.0,
    "weights": [1.0, 1.0],
    "tta_flip": True,
    "tta_flip_models": [True, True],
    "models": None,
}

if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as _f:
        _cfg = json.load(_f)
    _DEFAULTS.update(_cfg)

NC = _DEFAULTS["nc"]
CONF_THRESHOLD = _DEFAULTS["conf_threshold"]
MAX_PREDS = _DEFAULTS["max_predictions_per_image"]
WBF_IOU = _DEFAULTS["wbf_iou_threshold"]
WBF_SKIP = _DEFAULTS["wbf_skip_threshold"]
WEIGHTS = _DEFAULTS["weights"]
TTA_FLIP = _DEFAULTS["tta_flip"]
TTA_FLIP_MODELS = _DEFAULTS["tta_flip_models"]


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


def resolve_model_specs(config, script_dir: Path):
    raw_models = config.get("models")
    if isinstance(raw_models, list) and raw_models:
        specs = []
        for idx, model_cfg in enumerate(raw_models):
            if not isinstance(model_cfg, dict):
                continue
            file_name = model_cfg.get("file")
            if not file_name:
                continue
            specs.append(
                {
                    "name": model_cfg.get("name", chr(ord("A") + idx)),
                    "path": script_dir / file_name,
                    "imgsz": int(model_cfg.get("imgsz", 1280)),
                    "tta_flip": bool(model_cfg.get("tta_flip", config.get("tta_flip", False))),
                    "weight": float(model_cfg.get("weight", 1.0)),
                }
            )
        if specs:
            return specs

    legacy = []
    for idx, suffix in enumerate(["a", "b", "c"]):
        model_key = f"model_{suffix}"
        imgsz_key = f"imgsz_{suffix}"
        if model_key in config:
            legacy.append(
                {
                    "name": suffix.upper(),
                    "path": script_dir / config[model_key],
                    "imgsz": int(config.get(imgsz_key, 1280)),
                    "tta_flip": bool(
                        config.get("tta_flip_models", [config.get("tta_flip", False)])[idx]
                        if idx < len(config.get("tta_flip_models", []))
                        else config.get("tta_flip", False)
                    ),
                    "weight": float(config.get("weights", [1.0])[idx])
                    if idx < len(config.get("weights", []))
                    else 1.0,
                }
            )
    return legacy


MODEL_SPECS = resolve_model_specs(_DEFAULTS, SCRIPT_DIR)


def _run_inference(session, input_name, img_np, orig_shape, imgsz, conf_thresh, flip=False):
    """Run one ONNX model pass. Returns (boxes_norm, scores, labels) for WBF."""
    oh, ow = orig_shape
    inp = img_np[:, ::-1, :].copy() if flip else img_np

    padded, r, pad = letterbox(inp, imgsz)
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

    output = session.run(None, {input_name: blob})
    output = output[0][0].T

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

    pad_x, pad_y = pad
    x1 = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2 - pad_x) / r
    y1 = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2 - pad_y) / r
    x2 = (boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2 - pad_x) / r
    y2 = (boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2 - pad_y) / r

    if flip:
        x1_new = ow - x2
        x2_new = ow - x1
        x1 = x1_new
        x2 = x2_new

    boxes_norm = np.stack([
        np.clip(x1 / ow, 0, 1),
        np.clip(y1 / oh, 0, 1),
        np.clip(x2 / ow, 0, 1),
        np.clip(y2 / oh, 0, 1),
    ], axis=1)

    valid = (boxes_norm[:, 2] > boxes_norm[:, 0]) & (boxes_norm[:, 3] > boxes_norm[:, 1])
    return boxes_norm[valid], max_scores[valid], class_ids[valid]


def run_single_model(session, input_name, img_np, orig_shape, imgsz, conf_thresh, tta_flip=True):
    boxes, scores, labels = _run_inference(
        session, input_name, img_np, orig_shape, imgsz, conf_thresh, flip=False
    )
    if not tta_flip:
        return boxes, scores, labels

    boxes_f, scores_f, labels_f = _run_inference(
        session, input_name, img_np, orig_shape, imgsz, conf_thresh, flip=True
    )
    if len(boxes) == 0:
        return boxes_f, scores_f, labels_f
    if len(boxes_f) == 0:
        return boxes, scores, labels

    return (
        np.concatenate([boxes, boxes_f]),
        np.concatenate([scores, scores_f]),
        np.concatenate([labels, labels_f]),
    )


def load_sessions(model_specs, providers):
    sessions = []
    for spec in model_specs:
        path = spec["path"]
        if not path.exists():
            print(f"[WARN] Model {spec['name']} not found at {path}")
            continue
        try:
            session = ort.InferenceSession(str(path), providers=providers)
            sessions.append(
                {
                    **spec,
                    "session": session,
                    "input_name": session.get_inputs()[0].name,
                }
            )
            print(
                f"Model {spec['name']}: {path.name}, imgsz={spec['imgsz']}, "
                f"flip={spec['tta_flip']}, weight={spec['weight']}"
            )
        except Exception as e:
            print(f"[ERROR] Failed to load model {spec['name']}: {e}")
    return sessions


def predict_image_ensemble(sessions, img_np, orig_shape):
    oh, ow = orig_shape
    all_boxes = []
    all_scores = []
    all_labels = []
    weights = []

    for spec in sessions:
        boxes, scores, labels = run_single_model(
            spec["session"],
            spec["input_name"],
            img_np,
            orig_shape,
            spec["imgsz"],
            CONF_THRESHOLD,
            tta_flip=spec["tta_flip"],
        )
        all_boxes.append(boxes.tolist() if len(boxes) > 0 else [])
        all_scores.append(scores.tolist() if len(scores) > 0 else [])
        all_labels.append(labels.astype(int).tolist() if len(labels) > 0 else [])
        weights.append(spec["weight"])

    predictions = []
    if any(len(b) > 0 for b in all_boxes):
        boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
            all_boxes,
            all_scores,
            all_labels,
            weights=weights,
            iou_thr=WBF_IOU,
            skip_box_thr=CONF_THRESHOLD,
        )

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
            predictions.append({
                "category_id": cat_id,
                "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                "score": round(float(scores_fused[i]), 4),
            })
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sessions = load_sessions(MODEL_SPECS, providers)

    if not sessions:
        output_path.write_text("[]")
        return

    print(f"Ensemble: {len(sessions)} models, WBF IoU={WBF_IOU}, conf={CONF_THRESHOLD}")

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
            preds = predict_image_ensemble(sessions, img_np, img_np.shape[:2])
            for pred in preds:
                pred["image_id"] = image_id
            all_predictions.extend(preds)
        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")
            continue

    output_path.write_text(json.dumps(all_predictions))
    print(f"Wrote {len(all_predictions)} predictions for {len(image_files)} images")


if __name__ == "__main__":
    main()
