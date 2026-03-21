"""
Generate tiled training dataset from shelf images.

Splits by ORIGINAL IMAGE first, then tiles train originals only.
Val images remain full-size and untouched.

Usage:
  python data/prepare_tiles.py --tile-size 1536 --stride 768
  python data/prepare_tiles.py --tile-size 1280 --stride 640
"""
import argparse
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

from PIL import Image

from tiling import generate_tile_coords, clip_bbox_to_tile, bbox_to_yolo


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

ANNOTATIONS_PATH = RAW_DIR / "annotations.json"
IMAGES_DIR = RAW_DIR / "images"
VAL_SPLIT_PATH = DATA_DIR / "val_split.json"
OUTPUT_BASE = PROJECT_DIR / "datasets"

SEED = 42
EMPTY_TILE_CAP_RATIO = 0.20  # Cap empties at 20% of positive tile count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-size", type=int, default=1536)
    parser.add_argument("--stride", type=int, default=768)
    parser.add_argument("--all-data", action="store_true", help="Use all images for train (no val holdout)")
    args = parser.parse_args()

    random.seed(SEED)

    tile_size = args.tile_size
    stride = args.stride
    variant = f"tiled-{tile_size}" + ("-alldata" if args.all_data else "")

    # Load data
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)

    with open(VAL_SPLIT_PATH) as f:
        split = json.load(f)
        train_ids = set(split["train_ids"])
        val_ids = set(split["val_ids"])

    if args.all_data:
        train_ids = set(img["id"] for img in data["images"])
        val_ids = set(list(train_ids)[:10])  # Small val for YOLO compatibility

    # Build lookups
    id_to_img = {img["id"]: img for img in data["images"]}
    img_to_anns = defaultdict(list)
    for ann in data["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)

    categories = data["categories"]
    nc = len(categories)
    cat_names = {c["id"]: c["name"] for c in categories}

    # Output directory
    out_dir = OUTPUT_BASE / variant
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images" / "train").mkdir(parents=True)
    (out_dir / "labels" / "train").mkdir(parents=True)
    (out_dir / "images" / "val").mkdir(parents=True)
    (out_dir / "labels" / "val").mkdir(parents=True)

    print(f"Tiling: size={tile_size}, stride={stride}, variant={variant}")
    print(f"Images: {len(train_ids)} train, {len(val_ids)} val")

    # --- Tile train images ---
    tile_metadata = []
    positive_tiles = 0
    empty_tiles_all = []
    total_annotations = 0
    tile_idx = 0

    for img_id in sorted(train_ids):
        img_info = id_to_img[img_id]
        img_w = img_info["width"]
        img_h = img_info["height"]
        fname = img_info["file_name"]
        src_path = IMAGES_DIR / fname

        if not src_path.exists():
            continue

        try:
            img = Image.open(src_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Failed to open {fname}: {e}")
            continue

        anns = img_to_anns[img_id]
        tile_coords = generate_tile_coords(img_w, img_h, tile_size, stride)

        for tx, ty in tile_coords:
            # Determine actual tile dimensions (handle small images)
            actual_tw = min(tile_size, img_w - tx)
            actual_th = min(tile_size, img_h - ty)

            if actual_tw < tile_size or actual_th < tile_size:
                # For undersized tiles, pad or skip
                # For simplicity, we'll crop what we can
                pass

            # Clip annotations to this tile
            tile_labels = []
            for ann in anns:
                clipped, include = clip_bbox_to_tile(
                    ann["bbox"], tx, ty, tile_size
                )
                if include:
                    cx, cy, nw, nh = bbox_to_yolo(clipped, tile_size)
                    tile_labels.append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            tile_name = f"tile_{img_id:05d}_{tx:04d}_{ty:04d}"

            if len(tile_labels) > 0:
                # Positive tile — always keep
                # Crop and save
                crop = img.crop((tx, ty, tx + tile_size, ty + tile_size))
                crop.save(out_dir / "images" / "train" / f"{tile_name}.jpg", quality=95)
                (out_dir / "labels" / "train" / f"{tile_name}.txt").write_text("\n".join(tile_labels))

                tile_metadata.append({
                    "tile": tile_name,
                    "source_image": fname,
                    "source_id": img_id,
                    "tile_x": tx, "tile_y": ty,
                    "tile_size": tile_size,
                    "n_annotations": len(tile_labels),
                })

                positive_tiles += 1
                total_annotations += len(tile_labels)
            else:
                # Empty tile — collect for capped sampling later
                empty_tiles_all.append({
                    "img": img, "tx": tx, "ty": ty,
                    "tile_name": tile_name, "source": fname, "img_id": img_id,
                })

            tile_idx += 1

    # Sample empty tiles (capped at 20% of positive count)
    max_empty = int(positive_tiles * EMPTY_TILE_CAP_RATIO)
    random.shuffle(empty_tiles_all)
    sampled_empties = empty_tiles_all[:max_empty]

    for et in sampled_empties:
        crop = et["img"].crop((et["tx"], et["ty"], et["tx"] + tile_size, et["ty"] + tile_size))
        crop.save(out_dir / "images" / "train" / f"{et['tile_name']}.jpg", quality=95)
        (out_dir / "labels" / "train" / f"{et['tile_name']}.txt").write_text("")

        tile_metadata.append({
            "tile": et["tile_name"],
            "source_image": et["source"],
            "source_id": et["img_id"],
            "tile_x": et["tx"], "tile_y": et["ty"],
            "tile_size": tile_size,
            "n_annotations": 0,
        })

    # --- Copy val images at full resolution (untouched) ---
    val_ann_count = 0
    for img_id in sorted(val_ids):
        img_info = id_to_img[img_id]
        fname = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        src = IMAGES_DIR / fname
        if src.exists():
            shutil.copy2(src, out_dir / "images" / "val" / fname)

        # Write full-image YOLO labels for val
        anns = img_to_anns[img_id]
        lines = []
        for ann in anns:
            bx, by, bw, bh = ann["bbox"]
            cx = (bx + bw / 2) / img_w
            cy = (by + bh / 2) / img_h
            nw = bw / img_w
            nh = bh / img_h
            lines.append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            val_ann_count += 1

        label_name = Path(fname).stem + ".txt"
        (out_dir / "labels" / "val" / label_name).write_text("\n".join(lines))

    # Write data.yaml
    names_lines = [f'  {i}: "{cat_names[i]}"' for i in range(nc)]
    names_block = "names:\n" + "\n".join(names_lines) + "\n"
    yaml_content = f"""path: {out_dir}
train: images/train
val: images/val

nc: {nc}
{names_block}"""
    (out_dir / "data.yaml").write_text(yaml_content)

    # Write tile metadata
    with open(out_dir / "tile_metadata.json", "w") as f:
        json.dump(tile_metadata, f, indent=2)

    # Stats
    print(f"\n=== Dataset '{variant}' ===")
    print(f"  Train tiles (positive): {positive_tiles}")
    print(f"  Train tiles (empty, sampled): {len(sampled_empties)} / {len(empty_tiles_all)} total empty")
    print(f"  Train tiles (total): {positive_tiles + len(sampled_empties)}")
    print(f"  Train annotations: {total_annotations}")
    print(f"  Val images: {len(val_ids)} (full-size)")
    print(f"  Val annotations: {val_ann_count}")
    print(f"  Written to: {out_dir}")


if __name__ == "__main__":
    main()
