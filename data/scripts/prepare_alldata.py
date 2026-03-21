"""
Create an all-data YOLO dataset where all 248 images are in train.
Val uses a small subset (YOLO requires val to run) but all images
are in train for maximum training data.

Usage:
  python data/prepare_alldata.py
"""
import json
import shutil
from pathlib import Path
from collections import defaultdict


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

ANNOTATIONS_PATH = RAW_DIR / "annotations.json"
IMAGES_DIR = RAW_DIR / "images"
OUTPUT_BASE = PROJECT_DIR / "datasets"


def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def main():
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)

    all_ids = [img["id"] for img in data["images"]]
    # All images go to train; first 10 also serve as val (YOLO requires val)
    train_ids = all_ids
    val_ids = all_ids[:10]

    id_to_img = {img["id"]: img for img in data["images"]}
    img_to_anns = defaultdict(list)
    for ann in data["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)

    categories = data["categories"]
    nc = len(categories)
    cat_names = {c["id"]: c["name"] for c in categories}

    out_dir = OUTPUT_BASE / "full-class-alldata"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        for img_id in ids:
            img_info = id_to_img[img_id]
            img_w = img_info["width"]
            img_h = img_info["height"]
            fname = img_info["file_name"]

            src = IMAGES_DIR / fname
            dst = out_dir / "images" / split / fname
            if src.exists():
                shutil.copy2(src, dst)

            anns = img_to_anns[img_id]
            label_name = Path(fname).stem + ".txt"
            label_path = out_dir / "labels" / split / label_name

            lines = []
            for ann in anns:
                cx, cy, nw, nh = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
                lines.append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            label_path.write_text("\n".join(lines))

    names_lines = [f'  {i}: "{cat_names[i]}"' for i in range(nc)]
    names_block = "names:\n" + "\n".join(names_lines) + "\n"

    yaml_content = f"""path: {out_dir}
train: images/train
val: images/val

nc: {nc}
{names_block}"""

    (out_dir / "data.yaml").write_text(yaml_content)

    train_anns = sum(len(img_to_anns[i]) for i in train_ids)
    print(f"Dataset 'full-class-alldata' written to {out_dir}")
    print(f"  Train: {len(train_ids)} images, {train_anns} annotations (ALL data)")
    print(f"  Val: {len(val_ids)} images (subset, for YOLO compatibility)")


if __name__ == "__main__":
    main()
