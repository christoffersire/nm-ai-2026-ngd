"""
Prepare classifier training data from shelf image annotations.

Crops GT bounding boxes with padding and generates jittered variants
to simulate detector imperfection. Organizes into ultralytics classify
folder structure.

Usage:
  python train/prepare_classifier_data.py
  python train/prepare_classifier_data.py --jitter-count 3 --crop-size 256
"""
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict, Counter

from PIL import Image


# --- Paths (use Path.home() for portability between local and VM) ---
ANNOTATIONS_PATH = Path.home() / "Downloads" / "train" / "annotations.json"
IMAGES_DIR = Path.home() / "Downloads" / "train" / "images"
VAL_SPLIT_PATH = Path.home() / "nm-ai-2026-ngd" / "data" / "val_split.json"
OUTPUT_BASE = Path.home() / "nm-ai-2026-ngd" / "datasets" / "classifier"

SEED = 42
PADDING = 0.1  # 10% padding around GT bbox


def compute_iou_xywh(box_a, box_b):
    """IoU between two COCO [x, y, w, h] boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0, ix2 - ix) * max(0, iy2 - iy)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def crop_with_padding(img, bbox, padding=PADDING):
    """Crop bbox from image with padding, clamped to image bounds."""
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


def generate_jittered_crops(img, bbox, count=2, pos_jitter=0.05, scale_jitter=0.10,
                            min_iou=0.6, padding=PADDING):
    """
    Generate jittered crop variants around GT bbox.
    Simulates detector imperfection. Discards if IoU with GT < min_iou.
    """
    x, y, w, h = bbox
    crops = []

    for _ in range(count * 3):  # oversample to account for IoU filtering
        if len(crops) >= count:
            break

        # Random position shift
        dx = random.uniform(-pos_jitter, pos_jitter) * w
        dy = random.uniform(-pos_jitter, pos_jitter) * h

        # Random scale
        sw = random.uniform(1 - scale_jitter, 1 + scale_jitter)
        sh = random.uniform(1 - scale_jitter, 1 + scale_jitter)

        new_w = w * sw
        new_h = h * sh
        # Keep center roughly aligned
        new_x = x + dx + (w - new_w) / 2
        new_y = y + dy + (h - new_h) / 2

        jittered_bbox = [new_x, new_y, new_w, new_h]

        # Check IoU with original GT
        iou = compute_iou_xywh(bbox, jittered_bbox)
        if iou < min_iou:
            continue

        crop = crop_with_padding(img, jittered_bbox, padding=padding)
        if crop is not None and crop.size[0] > 2 and crop.size[1] > 2:
            crops.append(crop)

    return crops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-size", type=int, default=224, help="Output crop size")
    parser.add_argument("--jitter-count", type=int, default=2, help="Jittered variants per GT crop")
    parser.add_argument("--no-jitter", action="store_true", help="Skip jitter generation")
    args = parser.parse_args()

    random.seed(SEED)

    # Load data
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)

    with open(VAL_SPLIT_PATH) as f:
        split = json.load(f)
        train_ids = set(split["train_ids"])
        val_ids = set(split["val_ids"])

    # Build lookups
    id_to_img = {img["id"]: img for img in data["images"]}
    img_to_anns = defaultdict(list)
    for ann in data["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)

    categories = {c["id"]: c["name"] for c in data["categories"]}
    nc = len(categories)

    print(f"Loaded: {len(data['images'])} images, {len(data['annotations'])} annotations, {nc} categories")
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val")
    print(f"Crop size: {args.crop_size}, Jitter: {'off' if args.no_jitter else f'{args.jitter_count} per crop'}")

    # Clean output
    import shutil
    if OUTPUT_BASE.exists():
        shutil.rmtree(OUTPUT_BASE)

    # Create category directories
    for cat_id in range(nc):
        (OUTPUT_BASE / "train" / f"{cat_id:03d}").mkdir(parents=True, exist_ok=True)
        (OUTPUT_BASE / "val" / f"{cat_id:03d}").mkdir(parents=True, exist_ok=True)

    # Track stats
    train_counts = Counter()
    val_counts = Counter()
    jitter_counts = Counter()
    skipped = 0

    # Process images
    for img_info in data["images"]:
        img_id = img_info["id"]
        fname = img_info["file_name"]
        is_train = img_id in train_ids
        is_val = img_id in val_ids

        if not is_train and not is_val:
            continue

        img_path = IMAGES_DIR / fname
        if not img_path.exists():
            print(f"  [WARN] Missing image: {fname}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Failed to open {fname}: {e}")
            skipped += 1
            continue

        anns = img_to_anns[img_id]
        split_name = "train" if is_train else "val"

        for ann in anns:
            cat_id = ann["category_id"]
            ann_id = ann["id"]
            bbox = ann["bbox"]

            # Skip tiny boxes
            if bbox[2] < 3 or bbox[3] < 3:
                skipped += 1
                continue

            # GT crop
            crop = crop_with_padding(img, bbox)
            if crop is None:
                skipped += 1
                continue

            crop_resized = crop.resize((args.crop_size, args.crop_size), Image.LANCZOS)
            crop_name = f"crop_img{img_id:05d}_ann{ann_id:06d}.jpg"
            crop_resized.save(OUTPUT_BASE / split_name / f"{cat_id:03d}" / crop_name, quality=95)

            if is_train:
                train_counts[cat_id] += 1
            else:
                val_counts[cat_id] += 1

            # Jittered crops (train only)
            if is_train and not args.no_jitter:
                jittered = generate_jittered_crops(img, bbox, count=args.jitter_count)
                for j, jcrop in enumerate(jittered):
                    jcrop_resized = jcrop.resize((args.crop_size, args.crop_size), Image.LANCZOS)
                    jcrop_name = f"crop_img{img_id:05d}_ann{ann_id:06d}_jit{j}.jpg"
                    jcrop_resized.save(OUTPUT_BASE / split_name / f"{cat_id:03d}" / jcrop_name, quality=95)
                    jitter_counts[cat_id] += 1

    # Print statistics
    total_train = sum(train_counts.values())
    total_jitter = sum(jitter_counts.values())
    total_val = sum(val_counts.values())

    print(f"\n=== Dataset Statistics ===")
    print(f"Train GT crops:     {total_train}")
    print(f"Train jitter crops: {total_jitter}")
    print(f"Train total:        {total_train + total_jitter}")
    print(f"Val crops:          {total_val}")
    print(f"Skipped:            {skipped}")
    print(f"Categories used:    {len(train_counts)} train, {len(val_counts)} val")

    # Low-shot analysis
    low_shot = [cat_id for cat_id, count in train_counts.items() if count <= 5]
    zero_shot_val = [cat_id for cat_id in range(nc) if val_counts[cat_id] == 0]

    print(f"\n=== Class Balance ===")
    print(f"Low-shot classes (≤5 GT train crops): {len(low_shot)}")
    print(f"Zero-shot val classes:                {len(zero_shot_val)}")

    # Distribution buckets
    buckets = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float("inf"))]
    print(f"\nTrain GT crop distribution:")
    for lo, hi in buckets:
        count = sum(1 for c in train_counts.values() if lo <= c <= hi)
        label = f"{lo}-{hi}" if hi != float("inf") else f"{lo}+"
        print(f"  {label:>8}: {count} classes")

    print(f"\nDataset written to {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
