"""
Create a YOLO detection dataset that oversamples images containing
rare or historically high-corruption classes.

This is designed for training a specialist model that complements the
strong general detector/ensemble used in the current best submission.

The output dataset keeps one physical copy of each image/label, then
uses a duplicated train.txt manifest to oversample selected images.

Examples:
  python3 data/scripts/prepare_tail_boost.py
  python3 data/scripts/prepare_tail_boost.py --all-data --output-name full-class-tail-boost-alldata
  python3 data/scripts/prepare_tail_boost.py --annotations data/raw/annotations_v17.json
"""
import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = RAW_DIR / "images"
OUTPUT_BASE = PROJECT_DIR / "datasets"
DEFAULT_VAL_SPLIT_PATH = DATA_DIR / "val_split.json"
DEFAULT_METADATA_PATH = DATA_DIR / "metadata.json"


def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def choose_split(data, val_split_path: Path, all_data: bool, all_data_val_count: int):
    image_ids = [img["id"] for img in data["images"]]

    if all_data:
        if val_split_path.exists():
            val_ids = load_json(val_split_path)["val_ids"][:all_data_val_count]
        else:
            val_ids = image_ids[:all_data_val_count]
        train_ids = image_ids
        return train_ids, val_ids

    if not val_split_path.exists():
        raise FileNotFoundError(
            f"Missing val split at {val_split_path}. Run data/scripts/prepare_yolo.py first."
        )

    split = load_json(val_split_path)
    return split["train_ids"], split["val_ids"]


def load_metadata_by_name(metadata_path: Path):
    if not metadata_path.exists():
        return {}

    metadata = load_json(metadata_path)
    items = metadata.get("products") if isinstance(metadata, dict) and "products" in metadata else metadata
    by_name = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("product_name") or item.get("name") or item.get("display_name")
        if name:
            by_name[name] = item
    return by_name


def build_boost_table(
    data,
    metadata_by_name,
    rare_threshold: int,
    medium_threshold: int,
    rare_repeat: int,
    medium_repeat: int,
    high_correction_threshold: float,
    mid_correction_threshold: float,
    high_correction_repeat: int,
    mid_correction_repeat: int,
    min_correction_count: int,
    max_repeat: int,
):
    counts = Counter(ann["category_id"] for ann in data["annotations"])
    categories = data["categories"]

    boost_table = {}
    for cat in categories:
        cat_id = cat["id"]
        name = cat["name"]
        ann_count = counts.get(cat_id, 0)
        repeat = 1
        reasons = []

        if ann_count <= rare_threshold and ann_count > 0:
            repeat = max(repeat, rare_repeat)
            reasons.append(f"rare<={rare_threshold}")
        elif ann_count <= medium_threshold and ann_count > 0:
            repeat = max(repeat, medium_repeat)
            reasons.append(f"tail<={medium_threshold}")

        meta = metadata_by_name.get(name, {})
        corrected_count = int(meta.get("corrected_count", 0) or 0)
        meta_ann_count = int(meta.get("annotation_count", ann_count) or ann_count)
        correction_fraction = corrected_count / meta_ann_count if meta_ann_count else 0.0

        # Skip boosting unknown_product from corruption stats; it is already abundant.
        if name != "unknown_product" and corrected_count >= min_correction_count:
            if correction_fraction >= high_correction_threshold:
                repeat = max(repeat, high_correction_repeat)
                reasons.append(f"corr>={high_correction_threshold:.2f}")
            elif correction_fraction >= mid_correction_threshold:
                repeat = max(repeat, mid_correction_repeat)
                reasons.append(f"corr>={mid_correction_threshold:.2f}")

        boost_table[cat_id] = {
            "name": name,
            "ann_count": ann_count,
            "corrected_count": corrected_count,
            "correction_fraction": round(correction_fraction, 4),
            "repeat": min(repeat, max_repeat),
            "reasons": reasons,
        }

    return boost_table


def write_dataset(
    data,
    output_name: str,
    train_ids,
    val_ids,
    boost_table,
    min_boosted_annotations: int,
    min_boost_share: float,
    isolated_strong_repeat: int,
    strong_repeat_floor: int,
):
    out_dir = OUTPUT_BASE / output_name
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)

    id_to_img = {img["id"]: img for img in data["images"]}
    img_to_anns = defaultdict(list)
    for ann in data["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)

    categories = data["categories"]
    nc = len(categories)
    cat_names = {c["id"]: c["name"] for c in categories}

    train_manifest = []
    val_manifest = []
    image_repeat_report = []

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
            repeat = 1
            boost_sources = []
            boosted_count = 0
            strongest_repeat = 1
            for ann in anns:
                cx, cy, nw, nh = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
                lines.append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                class_boost = boost_table[ann["category_id"]]
                if class_boost["repeat"] > 1:
                    boosted_count += 1
                    strongest_repeat = max(strongest_repeat, class_boost["repeat"])
                    boost_sources.append(
                        {
                            "category_id": ann["category_id"],
                            "name": class_boost["name"],
                            "repeat": class_boost["repeat"],
                            "reasons": class_boost["reasons"],
                        }
                    )

            label_path.write_text("\n".join(lines))

            img_path = (out_dir / "images" / split / fname).resolve()
            if split == "train":
                boost_share = boosted_count / max(len(anns), 1)
                if strongest_repeat > 1:
                    if boosted_count >= min_boosted_annotations or boost_share >= min_boost_share:
                        repeat = strongest_repeat
                    elif strongest_repeat >= strong_repeat_floor:
                        repeat = max(1, isolated_strong_repeat)

                train_manifest.extend([str(img_path)] * repeat)
                image_repeat_report.append(
                    {
                        "image_id": img_id,
                        "file_name": fname,
                        "repeat": repeat,
                        "annotation_count": len(anns),
                        "boosted_annotation_count": boosted_count,
                        "boost_share": round(boost_share, 4),
                        "boost_sources": boost_sources,
                    }
                )
            else:
                val_manifest.append(str(img_path))

    names_lines = [f'  {i}: "{cat_names[i]}"' for i in range(nc)]
    names_block = "names:\n" + "\n".join(names_lines) + "\n"

    (out_dir / "train.txt").write_text("\n".join(train_manifest) + "\n")
    (out_dir / "val.txt").write_text("\n".join(val_manifest) + "\n")

    yaml_content = f"""path: {out_dir}
train: train.txt
val: val.txt

nc: {nc}
{names_block}"""
    (out_dir / "data.yaml").write_text(yaml_content)

    boosted_classes = [
        {
            "category_id": cat_id,
            **info,
        }
        for cat_id, info in boost_table.items()
        if info["repeat"] > 1
    ]
    boosted_classes.sort(key=lambda row: (-row["repeat"], row["ann_count"], row["category_id"]))

    image_repeat_report.sort(key=lambda row: (-row["repeat"], row["image_id"]))
    repeated_images = [row for row in image_repeat_report if row["repeat"] > 1]

    report = {
        "output_name": output_name,
        "train_image_count": len(train_ids),
        "val_image_count": len(val_ids),
        "train_manifest_entries": len(train_manifest),
        "oversample_factor": round(len(train_manifest) / max(len(train_ids), 1), 4),
        "boosted_class_count": len(boosted_classes),
        "repeated_image_count": len(repeated_images),
        "boosted_classes": boosted_classes,
        "repeated_images": repeated_images[:100],
    }
    (out_dir / "boost_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"Dataset '{output_name}' written to {out_dir}")
    print(f"  Train images: {len(train_ids)} unique -> {len(train_manifest)} manifest entries")
    print(f"  Val images:   {len(val_ids)}")
    print(f"  Boosted classes: {len(boosted_classes)}")
    print(f"  Repeated images: {len(repeated_images)}")
    if boosted_classes:
        print("  Top boosted classes:")
        for row in boosted_classes[:15]:
            reasons = ",".join(row["reasons"])
            print(
                f"    cat {row['category_id']:>3} x{row['repeat']} "
                f"(anns={row['ann_count']}, corr={row['corrected_count']}, frac={row['correction_fraction']:.2f}) "
                f"{row['name']} [{reasons}]"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        default=str(RAW_DIR / "annotations.json"),
        help="COCO annotations file to convert",
    )
    parser.add_argument(
        "--val-split",
        default=str(DEFAULT_VAL_SPLIT_PATH),
        help="Path to val_split.json",
    )
    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA_PATH),
        help="Path to metadata.json with corrected_count statistics",
    )
    parser.add_argument(
        "--output-name",
        default="full-class-tail-boost",
        help="Dataset directory name under datasets/",
    )
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="Use all images for train and a small val subset for YOLO compatibility",
    )
    parser.add_argument(
        "--all-data-val-count",
        type=int,
        default=10,
        help="Number of val images to keep when --all-data is set",
    )
    parser.add_argument("--rare-threshold", type=int, default=5)
    parser.add_argument("--medium-threshold", type=int, default=10)
    parser.add_argument("--rare-repeat", type=int, default=6)
    parser.add_argument("--medium-repeat", type=int, default=3)
    parser.add_argument("--high-correction-threshold", type=float, default=0.50)
    parser.add_argument("--mid-correction-threshold", type=float, default=0.30)
    parser.add_argument("--high-correction-repeat", type=int, default=4)
    parser.add_argument("--mid-correction-repeat", type=int, default=3)
    parser.add_argument("--min-correction-count", type=int, default=5)
    parser.add_argument("--max-repeat", type=int, default=6)
    parser.add_argument(
        "--min-boosted-annotations",
        type=int,
        default=3,
        help="Require at least this many boosted boxes for full image repeat",
    )
    parser.add_argument(
        "--min-boost-share",
        type=float,
        default=0.12,
        help="Or require this share of boosted boxes for full image repeat",
    )
    parser.add_argument(
        "--isolated-strong-repeat",
        type=int,
        default=2,
        help="Repeat for images with isolated but very strong boost signals",
    )
    parser.add_argument(
        "--strong-repeat-floor",
        type=int,
        default=4,
        help="Class repeat at or above this value can trigger isolated strong repeat",
    )
    args = parser.parse_args()

    annotations_path = Path(args.annotations).resolve()
    val_split_path = Path(args.val_split).resolve()
    metadata_path = Path(args.metadata).resolve()

    data = load_json(annotations_path)
    metadata_by_name = load_metadata_by_name(metadata_path)
    train_ids, val_ids = choose_split(
        data,
        val_split_path=val_split_path,
        all_data=args.all_data,
        all_data_val_count=args.all_data_val_count,
    )

    boost_table = build_boost_table(
        data=data,
        metadata_by_name=metadata_by_name,
        rare_threshold=args.rare_threshold,
        medium_threshold=args.medium_threshold,
        rare_repeat=args.rare_repeat,
        medium_repeat=args.medium_repeat,
        high_correction_threshold=args.high_correction_threshold,
        mid_correction_threshold=args.mid_correction_threshold,
        high_correction_repeat=args.high_correction_repeat,
        mid_correction_repeat=args.mid_correction_repeat,
        min_correction_count=args.min_correction_count,
        max_repeat=args.max_repeat,
    )

    write_dataset(
        data=data,
        output_name=args.output_name,
        train_ids=train_ids,
        val_ids=val_ids,
        boost_table=boost_table,
        min_boosted_annotations=args.min_boosted_annotations,
        min_boost_share=args.min_boost_share,
        isolated_strong_repeat=args.isolated_strong_repeat,
        strong_repeat_floor=args.strong_repeat_floor,
    )


if __name__ == "__main__":
    main()
