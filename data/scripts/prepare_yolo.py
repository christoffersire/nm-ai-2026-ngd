"""
Convert COCO annotations to YOLO format and create train/val split.

Produces two dataset variants:
  - full-class: all 356 categories preserved (for Pipeline A)
  - single-class: all categories mapped to 0 (for Pipeline B)

Output structure:
  datasets/
    full-class/
      images/train/  images/val/
      labels/train/  labels/val/
      data.yaml
    single-class/
      images/train/  images/val/
      labels/train/  labels/val/
      data.yaml
"""
import json
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict


# --- Paths ---
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

ANNOTATIONS_PATH = RAW_DIR / "annotations.json"
IMAGES_DIR = RAW_DIR / "images"
OUTPUT_BASE = PROJECT_DIR / "datasets"
VAL_SPLIT_PATH = DATA_DIR / "val_split.json"

VAL_RATIO = 0.2  # ~48 val images
SEED = 42


def load_annotations():
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)
    return data


def create_val_split(data: dict) -> tuple[list[int], list[int]]:
    """Image-level split preserving rare category representation."""
    random.seed(SEED)

    images = data["images"]
    annotations = data["annotations"]

    # Group annotations by image
    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann["image_id"]].append(ann)

    # Count categories per image for stratification
    img_ids = [img["id"] for img in images]
    n_val = max(1, int(len(img_ids) * VAL_RATIO))

    # Simple stratified approach: ensure rare categories appear in train
    # Find images that are the sole source for rare categories
    cat_to_images = defaultdict(set)
    for ann in annotations:
        cat_to_images[ann["category_id"]].add(ann["image_id"])

    # Images that are sole source for any category must be in train
    must_train = set()
    for cat_id, img_set in cat_to_images.items():
        if len(img_set) == 1:
            must_train.update(img_set)

    # Remaining images can be split
    available_for_val = [img_id for img_id in img_ids if img_id not in must_train]
    random.shuffle(available_for_val)

    val_ids = set(available_for_val[:n_val])
    train_ids = [img_id for img_id in img_ids if img_id not in val_ids]
    val_ids = list(val_ids)

    print(f"Split: {len(train_ids)} train, {len(val_ids)} val")
    print(f"  Must-train images (sole source for a category): {len(must_train)}")

    return train_ids, val_ids


def coco_to_yolo_bbox(bbox, img_w, img_h):
    """Convert COCO [x, y, w, h] to YOLO [cx, cy, w, h] normalized."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def write_dataset(data, train_ids, val_ids, variant: str, single_class: bool = False):
    """Write YOLO-format dataset."""
    out_dir = OUTPUT_BASE / variant
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)

    # Build lookups
    id_to_img = {img["id"]: img for img in data["images"]}
    img_to_anns = defaultdict(list)
    for ann in data["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)

    categories = data["categories"]
    nc = len(categories)
    cat_names = {c["id"]: c["name"] for c in categories}

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        for img_id in ids:
            img_info = id_to_img[img_id]
            img_w = img_info["width"]
            img_h = img_info["height"]
            fname = img_info["file_name"]

            # Copy image
            src = IMAGES_DIR / fname
            dst = out_dir / "images" / split / fname
            if src.exists():
                shutil.copy2(src, dst)

            # Write labels
            anns = img_to_anns[img_id]
            label_name = Path(fname).stem + ".txt"
            label_path = out_dir / "labels" / split / label_name

            lines = []
            for ann in anns:
                cat_id = 0 if single_class else ann["category_id"]
                cx, cy, nw, nh = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
                lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            label_path.write_text("\n".join(lines))

    # Write data.yaml
    if single_class:
        names_block = "names:\n  0: product\n"
        nc_val = 1
    else:
        names_lines = [f"  {i}: \"{cat_names[i]}\"" for i in range(nc)]
        names_block = "names:\n" + "\n".join(names_lines) + "\n"
        nc_val = nc

    yaml_content = f"""path: {out_dir}
train: images/train
val: images/val

nc: {nc_val}
{names_block}"""

    (out_dir / "data.yaml").write_text(yaml_content)
    print(f"Dataset '{variant}' written to {out_dir}")
    print(f"  nc={nc_val}, train={len(train_ids)} images, val={len(val_ids)} images")


def main():
    data = load_annotations()

    print(f"Loaded: {len(data['images'])} images, {len(data['annotations'])} annotations, {len(data['categories'])} categories")

    # Create and save val split
    train_ids, val_ids = create_val_split(data)

    split_info = {"train_ids": sorted(train_ids), "val_ids": sorted(val_ids), "seed": SEED}
    VAL_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VAL_SPLIT_PATH.write_text(json.dumps(split_info, indent=2))
    print(f"Val split saved to {VAL_SPLIT_PATH}")

    # Write both variants
    write_dataset(data, train_ids, val_ids, "full-class", single_class=False)
    write_dataset(data, train_ids, val_ids, "single-class", single_class=True)


if __name__ == "__main__":
    main()
