"""
Data cleaning pipeline: fix corrupted category_id labels in training annotations.

Uses a trained ONNX detector to predict on all training images, then flags
annotations where the detector confidently disagrees with the GT label.
Cross-references with metadata corrected_count to calibrate corrections.

Usage:
  python data/clean_annotations.py                    # Dry run (report only)
  python data/clean_annotations.py --apply             # Write corrected annotations
  python data/clean_annotations.py --verify            # Visual verification of changes
"""
import argparse
import json
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
import onnxruntime as ort


# --- Paths ---
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

ANNOTATIONS_PATH = RAW_DIR / "annotations.json"
METADATA_PATH = DATA_DIR / "metadata.json"
IMAGE_DIR = RAW_DIR / "images"
MODEL_PATH = PROJECT_DIR / "weights/v3-1280.onnx"
OUTPUT_PATH = RAW_DIR / "annotations_cleaned.json"

IMGSZ = 1280
CONF_THRESHOLD = 0.25
IOU_MATCH_THRESHOLD = 0.5


def letterbox(img_np, new_shape=1280):
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


def compute_iou(box_a, box_b):
    """IoU between two [x, y, w, h] boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0, ix2 - ix) * max(0, iy2 - iy)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0


def run_detector(session, input_name, img_np):
    """Run ONNX detector, return list of (bbox_xywh, class_id, confidence)."""
    oh, ow = img_np.shape[:2]
    padded, r, (pad_x, pad_y) = letterbox(img_np, IMGSZ)
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

    output = session.run(None, {input_name: blob})
    output = output[0][0].T  # (num_boxes, 4+nc)

    boxes_cxcywh = output[:, :4]
    class_scores = output[:, 4:]
    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    mask = max_scores >= CONF_THRESHOLD
    boxes_cxcywh = boxes_cxcywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_cxcywh) == 0:
        return []

    # Convert to original image coordinates [x, y, w, h]
    cx = boxes_cxcywh[:, 0]
    cy = boxes_cxcywh[:, 1]
    w = boxes_cxcywh[:, 2]
    h = boxes_cxcywh[:, 3]
    x1 = (cx - w / 2 - pad_x) / r
    y1 = (cy - h / 2 - pad_y) / r
    bw = w / r
    bh = h / r

    results = []
    for i in range(len(x1)):
        results.append((
            [float(x1[i]), float(y1[i]), float(bw[i]), float(bh[i])],
            int(class_ids[i]),
            float(max_scores[i]),
        ))
    return results


def match_predictions_to_gt(predictions, gt_annotations):
    """Match each GT annotation to the best overlapping prediction.
    Returns list of (ann, best_pred_class, best_pred_conf, best_iou)."""
    matches = []
    for ann in gt_annotations:
        gt_bbox = ann["bbox"]
        best_iou = 0
        best_pred_cls = -1
        best_pred_conf = 0

        for pred_bbox, pred_cls, pred_conf in predictions:
            iou = compute_iou(gt_bbox, pred_bbox)
            if iou > best_iou:
                best_iou = iou
                best_pred_cls = pred_cls
                best_pred_conf = pred_conf

        matches.append({
            "ann_id": ann["id"],
            "image_id": ann["image_id"],
            "gt_cat": ann["category_id"],
            "gt_bbox": ann["bbox"],
            "pred_cat": best_pred_cls,
            "pred_conf": best_pred_conf,
            "iou": best_iou,
        })
    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write corrected annotations")
    parser.add_argument("--verify", action="store_true", help="Generate visual verification")
    parser.add_argument("--model", default=str(MODEL_PATH), help="ONNX model path")
    parser.add_argument("--min-conf", type=float, default=0.4,
                        help="Minimum detector confidence to consider a correction")
    args = parser.parse_args()

    # Load data
    print("Loading annotations and metadata...")
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)
    with open(METADATA_PATH) as f:
        meta = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    cat_name_to_id = {c["name"]: c["id"] for c in data["categories"]}
    id_to_img = {img["id"]: img for img in data["images"]}

    # Build corrected_count lookup
    corrected_by_cat = {}
    meta_names = set()
    for p in meta["products"]:
        meta_names.add(p["product_name"])
        if p["product_name"] in cat_name_to_id:
            cid = cat_name_to_id[p["product_name"]]
            corrected_by_cat[cid] = p["corrected_count"]

    ann_counts = Counter(a["category_id"] for a in data["annotations"])

    # Group annotations by image
    ann_by_img = defaultdict(list)
    for a in data["annotations"]:
        ann_by_img[a["image_id"]].append(a)

    # Load ONNX model
    print(f"Loading model: {args.model}")
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(args.model, providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"Model loaded. Input: {input_name}")

    # Run inference on all images
    all_matches = []
    image_ids = sorted(ann_by_img.keys())
    total_images = len(image_ids)

    print(f"\nRunning inference on {total_images} images...")
    t0 = time.time()

    for idx, img_id in enumerate(image_ids):
        img_info = id_to_img[img_id]
        img_path = IMAGE_DIR / img_info["file_name"]
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        predictions = run_detector(session, input_name, img_np)
        matches = match_predictions_to_gt(predictions, ann_by_img[img_id])
        all_matches.extend(matches)

        if (idx + 1) % 10 == 0 or idx == total_images - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (total_images - idx - 1) / rate
            print(f"  [{idx+1}/{total_images}] {rate:.1f} img/s, ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nInference complete: {len(all_matches)} annotations checked in {elapsed:.0f}s")

    # Find disagreements
    disagreements = []
    for m in all_matches:
        if (m["iou"] >= IOU_MATCH_THRESHOLD and
            m["pred_cat"] != m["gt_cat"] and
            m["pred_conf"] >= args.min_conf and
            m["pred_cat"] >= 0):
            disagreements.append(m)

    disagreements.sort(key=lambda x: -x["pred_conf"])

    print(f"\nDisagreements (IoU≥{IOU_MATCH_THRESHOLD}, conf≥{args.min_conf}): {len(disagreements)}")

    # Analyze by GT category
    changes_from = Counter()  # GT cat that detector disagrees with
    changes_to = Counter()    # What detector thinks it should be
    for d in disagreements:
        changes_from[d["gt_cat"]] += 1
        changes_to[d["pred_cat"]] += 1

    print(f"\nTop 20 categories losing annotations (detector disagrees with GT):")
    print(f"{'Cat':>4} {'#Disagree':>9} {'Corr_Count':>10} {'Name'}")
    print("-" * 80)
    for cat_id, count in changes_from.most_common(20):
        corr = corrected_by_cat.get(cat_id, "N/A")
        name = cat_id_to_name.get(cat_id, "?")
        print(f"{cat_id:>4} {count:>9} {str(corr):>10} {name[:50]}")

    print(f"\nTop 20 categories gaining annotations (detector thinks these are correct):")
    print(f"{'Cat':>4} {'#Gaining':>9} {'Deficit':>8} {'Name'}")
    print("-" * 80)
    for cat_id, count in changes_to.most_common(20):
        name = cat_id_to_name.get(cat_id, "?")
        print(f"{cat_id:>4} {count:>9} {'':>8} {name[:50]}")

    # Show sample corrections
    print(f"\nTop 30 highest-confidence corrections:")
    print(f"{'Conf':>5} {'IoU':>5} {'GT':>4} {'->':>3} {'Pred':>4}  GT_Name -> Pred_Name")
    print("-" * 100)
    for d in disagreements[:30]:
        gt_name = cat_id_to_name.get(d["gt_cat"], "?")[:35]
        pred_name = cat_id_to_name.get(d["pred_cat"], "?")[:35]
        img_name = id_to_img[d["image_id"]]["file_name"]
        print(f"{d['pred_conf']:5.3f} {d['iou']:5.3f} {d['gt_cat']:>4} ->  {d['pred_cat']:>4}  {gt_name} -> {pred_name}  [{img_name}]")

    # Save disagreements for inspection
    disagree_path = DATA_DIR / "mislabel_candidates.json"
    with open(disagree_path, "w") as f:
        json.dump(disagreements, f, indent=2)
    print(f"\nAll {len(disagreements)} candidates saved to {disagree_path}")

    # Apply corrections
    if args.apply:
        print(f"\n{'='*60}")
        print("APPLYING CORRECTIONS")
        print(f"{'='*60}")

        # Build correction map: ann_id -> new_category_id
        corrections = {}
        for d in disagreements:
            corrections[d["ann_id"]] = d["pred_cat"]

        # Create corrected annotations
        corrected_data = json.loads(json.dumps(data))  # deep copy
        n_changed = 0
        for ann in corrected_data["annotations"]:
            if ann["id"] in corrections:
                old_cat = ann["category_id"]
                new_cat = corrections[ann["id"]]
                ann["category_id"] = new_cat
                n_changed += 1

        print(f"Changed {n_changed} annotations")

        # Verify counts shifted in right direction
        new_counts = Counter(a["category_id"] for a in corrected_data["annotations"])
        print(f"\nCategory count changes (top 15 affected):")
        all_changed_cats = set(changes_from.keys()) | set(changes_to.keys())
        changes = []
        for cid in all_changed_cats:
            old = ann_counts.get(cid, 0)
            new = new_counts.get(cid, 0)
            if old != new:
                changes.append((cid, old, new, new - old))
        changes.sort(key=lambda x: abs(x[3]), reverse=True)
        for cid, old, new, diff in changes[:15]:
            name = cat_id_to_name.get(cid, "?")[:40]
            corr = corrected_by_cat.get(cid, "?")
            print(f"  cat {cid:>3}: {old:>4} -> {new:>4} ({diff:>+4}) corr_count={corr}  {name}")

        # Write output
        with open(OUTPUT_PATH, "w") as f:
            json.dump(corrected_data, f)
        print(f"\nCorrected annotations written to: {OUTPUT_PATH}")

    # Visual verification
    if args.verify:
        _generate_verification(disagreements[:20], data, id_to_img, cat_id_to_name)


def _generate_verification(samples, data, id_to_img, cat_id_to_name):
    """Generate visual verification grid of proposed corrections."""
    from PIL import ImageDraw

    out_dir = Path("/tmp/verification_samples")
    out_dir.mkdir(exist_ok=True)

    n = min(len(samples), 20)
    cols = 5
    rows = (n + cols - 1) // cols
    cell_w, cell_h = 220, 250
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    for i, d in enumerate(samples[:n]):
        img_info = id_to_img[d["image_id"]]
        img_path = IMAGE_DIR / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")
        x, y, w, h = d["gt_bbox"]
        crop = img.crop((x, y, x + w, y + h)).resize((180, 180), Image.LANCZOS)

        col = i % cols
        row = i // cols
        px = col * cell_w + 10
        py = row * cell_h + 10
        grid.paste(crop, (px, py))

        gt_name = cat_id_to_name.get(d["gt_cat"], "?")[:25]
        pred_name = cat_id_to_name.get(d["pred_cat"], "?")[:25]
        draw.text((px, py + 185), f"GT: {gt_name}", fill=(200, 0, 0))
        draw.text((px, py + 200), f"->: {pred_name}", fill=(0, 128, 0))
        draw.text((px, py + 215), f"conf={d['pred_conf']:.2f} iou={d['iou']:.2f}", fill=(100, 100, 100))
        # Red border = correction proposed
        draw.rectangle([px-1, py-1, px+181, py+181], outline=(200, 0, 0), width=2)

    out_path = out_dir / "proposed_corrections.png"
    grid.save(out_path)
    print(f"\nVerification grid saved to: {out_path}")


if __name__ == "__main__":
    main()
