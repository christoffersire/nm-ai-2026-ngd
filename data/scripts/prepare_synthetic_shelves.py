"""
Generate synthetic shelf training images by compositing product images
onto real shelf backgrounds.

For each training image, extracts shelf regions and places product images
at realistic positions and scales. Generates YOLO-format labels.

Usage:
  python data/prepare_synthetic_shelves.py
  python data/prepare_synthetic_shelves.py --num-per-category 5 --output datasets/synthetic
"""
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

ANNOTATIONS_PATH = RAW_DIR / "annotations.json"
IMAGES_DIR = RAW_DIR / "images"
PRODUCT_IMAGES_DIR = DATA_DIR / "ngdata_products_nobg"
CATEGORY_MAPPING_PATH = DATA_DIR / "category_mapping.json"
VAL_SPLIT_PATH = DATA_DIR / "val_split.json"
OUTPUT_BASE = PROJECT_DIR / "datasets"

SEED = 42


def load_shelf_backgrounds(data, train_ids):
    """Extract shelf background crops from training images."""
    id_to_img = {img["id"]: img for img in data["images"]}
    backgrounds = []

    for img_id in sorted(train_ids):
        img_info = id_to_img[img_id]
        fname = img_info["file_name"]
        src_path = IMAGES_DIR / fname
        if not src_path.exists():
            continue
        try:
            img = Image.open(src_path).convert("RGB")
            img_w, img_h = img.size
            # Take horizontal strips at different heights
            strip_h = 300
            for y in range(0, img_h - strip_h, strip_h):
                strip = img.crop((0, y, img_w, y + strip_h))
                backgrounds.append(strip)
        except Exception:
            continue

    return backgrounds


def get_product_image(cat_id, product_code):
    """Load product image for a category (background-removed PNG)."""
    path = PRODUCT_IMAGES_DIR / f"cat{cat_id:03d}_{product_code}.png"
    if path.exists():
        return Image.open(path).convert("RGBA")
    # Fallback to jpg
    path_jpg = PRODUCT_IMAGES_DIR / f"cat{cat_id:03d}_{product_code}.jpg"
    if path_jpg.exists():
        return Image.open(path_jpg).convert("RGBA")
    return None


def apply_shelf_transform(product_img, target_h):
    """Transform product image to look like it's on a shelf."""
    # Resize to target height maintaining aspect ratio
    w, h = product_img.size
    scale = target_h / h
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    product_img = product_img.resize((new_w, new_h), Image.LANCZOS)

    # Convert to RGB for transforms, keep alpha
    rgb = product_img.convert("RGB")

    # Random brightness (0.7-1.1 to simulate shelf lighting)
    brightness = ImageEnhance.Brightness(rgb)
    rgb = brightness.enhance(random.uniform(0.7, 1.1))

    # Random contrast (0.8-1.1)
    contrast = ImageEnhance.Contrast(rgb)
    rgb = contrast.enhance(random.uniform(0.85, 1.1))

    # Slight saturation reduction (shelf photos are less vivid than studio)
    color = ImageEnhance.Color(rgb)
    rgb = color.enhance(random.uniform(0.7, 1.0))

    # Slight blur (shelf photos aren't as sharp as studio)
    if random.random() < 0.5:
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

    # Reconstruct with alpha
    if product_img.mode == "RGBA":
        r, g, b = rgb.split()
        a = product_img.split()[3]
        return Image.merge("RGBA", (r, g, b, a))
    return rgb


def create_synthetic_image(background, products_to_place, target_size=1280):
    """
    Create a synthetic shelf image by placing products on a background strip.

    Returns (image, labels) where labels are YOLO format lines.
    """
    bg_w, bg_h = background.size

    # Resize background to target width
    scale = target_size / bg_w
    new_bg_h = max(1, int(bg_h * scale))
    bg = background.resize((target_size, new_bg_h), Image.LANCZOS)

    # Create square canvas
    canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    # Place background strip in the middle
    y_offset = (target_size - new_bg_h) // 2
    canvas.paste(bg, (0, y_offset))

    labels = []
    x_cursor = random.randint(10, 50)  # Start position

    for cat_id, product_img in products_to_place:
        if x_cursor >= target_size - 50:
            break

        # Target product height: 40-80% of background strip height
        target_h = int(new_bg_h * random.uniform(0.4, 0.8))
        transformed = apply_shelf_transform(product_img, target_h)

        pw, ph = transformed.size

        # Place at current x position, vertically centered on shelf
        place_y = y_offset + (new_bg_h - ph) // 2 + random.randint(-10, 10)
        place_y = max(0, min(target_size - ph, place_y))
        place_x = x_cursor

        if place_x + pw > target_size:
            break

        # Paste product
        if transformed.mode == "RGBA":
            canvas.paste(transformed, (place_x, place_y), transformed)
        else:
            canvas.paste(transformed, (place_x, place_y))

        # Generate YOLO label
        cx = (place_x + pw / 2) / target_size
        cy = (place_y + ph / 2) / target_size
        nw = pw / target_size
        nh = ph / target_size

        # Clamp to [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        nw = max(0, min(1, nw))
        nh = max(0, min(1, nh))

        if nw > 0.01 and nh > 0.01:
            labels.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        # Advance cursor with small gap
        x_cursor += pw + random.randint(2, 15)

    return canvas, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-per-category", type=int, default=5,
                        help="Synthetic images per rare category")
    parser.add_argument("--rare-threshold", type=int, default=10,
                        help="Categories with <= this many annotations get synthetic data")
    parser.add_argument("--output", default="synthetic-shelves",
                        help="Output dataset name")
    parser.add_argument("--target-size", type=int, default=1280)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    # Load annotations
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)

    with open(VAL_SPLIT_PATH) as f:
        split = json.load(f)
        train_ids = set(split["train_ids"])

    with open(CATEGORY_MAPPING_PATH) as f:
        cat_mapping = json.load(f)

    # Count annotations per category
    cat_counts = defaultdict(int)
    for ann in data["annotations"]:
        cat_counts[ann["category_id"]] += 1

    # Find rare categories with product images
    rare_cats = []
    for cat_id_str, info in cat_mapping.items():
        cat_id = int(cat_id_str)
        count = cat_counts.get(cat_id, 0)
        code = info.get("product_code", "")
        if count <= args.rare_threshold and code and code.isdigit():
            product_img = get_product_image(cat_id, code)
            if product_img is not None:
                rare_cats.append((cat_id, code, count, product_img))

    # Also collect ALL product images for diverse placement
    all_product_imgs = {}
    for cat_id_str, info in cat_mapping.items():
        cat_id = int(cat_id_str)
        code = info.get("product_code", "")
        if code and code.isdigit():
            pimg = get_product_image(cat_id, code)
            if pimg is not None:
                all_product_imgs[cat_id] = pimg

    print(f"Rare categories (≤{args.rare_threshold} annotations): {len(rare_cats)}")
    print(f"Total categories with product images: {len(all_product_imgs)}")

    # Load shelf backgrounds
    print("Loading shelf backgrounds...")
    backgrounds = load_shelf_backgrounds(data, train_ids)
    print(f"  {len(backgrounds)} background strips extracted")

    # Output directory
    out_dir = OUTPUT_BASE / args.output
    (out_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)

    # Generate synthetic images
    total_images = 0
    total_annotations = 0
    all_cat_ids = list(all_product_imgs.keys())

    for cat_id, code, count, product_img in rare_cats:
        for i in range(args.num_per_category):
            # Pick random background
            bg = random.choice(backgrounds).copy()

            # Place the target rare product + some random other products
            products_to_place = [(cat_id, product_img)]

            # Add 3-8 random other products for context
            n_others = random.randint(3, 8)
            other_cats = random.sample(all_cat_ids, min(n_others, len(all_cat_ids)))
            for other_cat in other_cats:
                if other_cat != cat_id:
                    products_to_place.append((other_cat, all_product_imgs[other_cat]))

            # Shuffle placement order
            random.shuffle(products_to_place)

            # Create synthetic image
            synth_img, labels = create_synthetic_image(
                bg, products_to_place, target_size=args.target_size
            )

            if len(labels) == 0:
                continue

            # Save
            img_name = f"synth_cat{cat_id:03d}_{i:02d}"
            synth_img.save(out_dir / "images" / "train" / f"{img_name}.jpg", quality=90)
            (out_dir / "labels" / "train" / f"{img_name}.txt").write_text("\n".join(labels))

            total_images += 1
            total_annotations += len(labels)

    print(f"\n=== Synthetic Dataset ===")
    print(f"  Images: {total_images}")
    print(f"  Annotations: {total_annotations}")
    print(f"  Written to: {out_dir}")


if __name__ == "__main__":
    main()
