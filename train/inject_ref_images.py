"""
Inject reference product images as training data for low-shot categories.

For categories with fewer than --min-shots GT crops in the classifier dataset,
this script finds the corresponding product reference images (front/main/back etc.)
from NM_NGD_product_images/ and adds augmented variants into the classifier
train folder so the model has something to learn from.

Augmentations applied to each reference image:
  - Original resized
  - Horizontal flip
  - Slight rotation (±10°)
  - Brightness/contrast jitter (×2 variants)
  - Random crop + resize (×2 variants)

This bridges the domain gap between clean product photos (white background)
and real shelf crops (cluttered background, lighting variation).

Usage:
  python train/inject_ref_images.py                    # threshold: ≤10 GT crops
  python train/inject_ref_images.py --min-shots 20     # more aggressive
  python train/inject_ref_images.py --dry-run          # just print what would happen
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter

try:
    from PIL import Image, ImageEnhance, ImageOps
except ImportError:
    raise SystemExit("Install Pillow: pip install Pillow")

# ─── Paths ────────────────────────────────────────────────────────────────────

ANNOTATIONS_PATH = Path.home() / "Downloads" / "train" / "annotations.json"
PRODUCT_IMAGES   = Path.home() / "Downloads" / "NM_NGD_product_images"
CATEGORY_MAPPING = Path(__file__).parent.parent / "data" / "category_mapping.json"
CLASSIFIER_DIR   = Path(__file__).parent.parent / "datasets" / "classifier"

# Order of preference for reference images
REF_PRIORITY = ["front.jpg", "main.jpg", "back.jpg", "left.jpg", "right.jpg", "top.jpg"]

CROP_SIZE = 224
SEED      = 42


# ─── Augmentations ────────────────────────────────────────────────────────────

def augment(img: Image.Image, crop_size: int) -> list[Image.Image]:
    """Return a list of augmented variants of img, all resized to crop_size."""
    imgs = []
    base = img.convert("RGB")

    def _save(i):
        imgs.append(i.resize((crop_size, crop_size), Image.LANCZOS))

    # 1. Original
    _save(base)

    # 2. Horizontal flip
    _save(ImageOps.mirror(base))

    # 3. Rotation ±10°
    for angle in (-10, 10):
        rotated = base.rotate(angle, expand=False, fillcolor=(128, 128, 128))
        _save(rotated)

    # 4. Brightness jitter
    for factor in (0.75, 1.30):
        _save(ImageEnhance.Brightness(base).enhance(factor))

    # 5. Contrast jitter
    for factor in (0.80, 1.25):
        _save(ImageEnhance.Contrast(base).enhance(factor))

    # 6. Random crop (80–95% of image, random position)
    w, h = base.size
    for _ in range(2):
        scale = random.uniform(0.80, 0.95)
        cw = int(w * scale)
        ch = int(h * scale)
        x0 = random.randint(0, w - cw)
        y0 = random.randint(0, h - ch)
        _save(base.crop((x0, y0, x0 + cw, y0 + ch)))

    return imgs


# ─── Helpers ──────────────────────────────────────────────────────────────────

def ref_images_for_code(product_code: str) -> list[Path]:
    """Return all available reference images for a product code, priority order."""
    if not product_code:
        return []
    folder = PRODUCT_IMAGES / product_code
    if not folder.exists():
        return []
    ordered = []
    for name in REF_PRIORITY:
        p = folder / name
        if p.exists():
            ordered.append(p)
    # Add any remaining jpgs not in priority list
    for p in sorted(folder.glob("*.jpg")):
        if p not in ordered:
            ordered.append(p)
    return ordered


def count_existing(train_dir: Path, cat_id: int) -> int:
    folder = train_dir / f"{cat_id:03d}"
    if not folder.exists():
        return 0
    return len(list(folder.glob("*.jpg")))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inject reference images for low-shot categories")
    parser.add_argument("--min-shots", type=int, default=10,
                        help="Inject for categories with fewer than this many train crops (default: 10)")
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE,
                        help=f"Output image size (default: {CROP_SIZE})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without writing files")
    args = parser.parse_args()

    random.seed(SEED)

    train_dir = CLASSIFIER_DIR / "train"
    if not train_dir.exists():
        raise SystemExit(
            f"Classifier train dir not found: {train_dir}\n"
            "Run prepare_classifier_data.py first."
        )

    # Load annotations to get category list and counts
    print("Loading annotations …")
    data = json.load(open(ANNOTATIONS_PATH))
    categories = {c["id"]: c["name"] for c in data["categories"]}
    ann_counts = Counter(a["category_id"] for a in data["annotations"])

    # Load category → product_code mapping
    cat_to_code = {}
    if CATEGORY_MAPPING.exists():
        raw = json.load(open(CATEGORY_MAPPING))
        for cat_id_str, info in raw.items():
            code = info.get("product_code")
            if code:
                cat_to_code[int(cat_id_str)] = code

    # Find low-shot categories
    low_shot = []
    for cat_id in sorted(categories.keys()):
        existing = count_existing(train_dir, cat_id)
        if existing < args.min_shots:
            code = cat_to_code.get(cat_id)
            ref_imgs = ref_images_for_code(code) if code else []
            low_shot.append({
                "cat_id":    cat_id,
                "name":      categories[cat_id],
                "code":      code,
                "existing":  existing,
                "ref_imgs":  ref_imgs,
            })

    print(f"\nKategorier med < {args.min_shots} treningscropper: {len(low_shot)}")
    print(f"  Har referansebilder:    {sum(1 for x in low_shot if x['ref_imgs'])}")
    print(f"  Ingen referansebilder:  {sum(1 for x in low_shot if not x['ref_imgs'])}")

    # Inject
    injected_cats = 0
    injected_imgs = 0
    skipped_no_ref = []
    skipped_no_code = []

    for item in low_shot:
        cat_id   = item["cat_id"]
        name     = item["name"]
        code     = item["code"]
        existing = item["existing"]
        ref_imgs = item["ref_imgs"]

        if not code:
            skipped_no_code.append(f"  cat#{cat_id:3d} (existing={existing}) {name}")
            continue
        if not ref_imgs:
            skipped_no_ref.append(f"  cat#{cat_id:3d} (existing={existing}, code={code}) {name}")
            continue

        out_dir = train_dir / f"{cat_id:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cat_injected = 0
        for ref_path in ref_imgs:
            try:
                img = Image.open(ref_path)
            except Exception as e:
                print(f"  [WARN] Cannot open {ref_path}: {e}")
                continue

            variants = augment(img, args.crop_size)
            for v_idx, variant in enumerate(variants):
                fname = f"ref_{ref_path.stem}_v{v_idx:02d}.jpg"
                out_path = out_dir / fname
                if not args.dry_run:
                    variant.save(out_path, quality=92)
                cat_injected += 1

        print(f"  cat#{cat_id:3d}  {name[:45]:<45}  "
              f"was={existing:3d}  +{cat_injected} ref imgs  "
              f"({len(ref_imgs)} source files × {len(variants)} augments)")

        injected_cats += 1
        injected_imgs += cat_injected

    # Summary
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}=== Ferdig ===")
    print(f"  Kategorier boosted:    {injected_cats}")
    print(f"  Bilder injisert:       {injected_imgs}")

    if skipped_no_code:
        print(f"\n  Hoppet over (ingen product_code, {len(skipped_no_code)} stk):")
        for s in skipped_no_code:
            print(s)

    if skipped_no_ref:
        print(f"\n  Hoppet over (ingen referansebilder i mappen, {len(skipped_no_ref)} stk):")
        for s in skipped_no_ref:
            print(s)

    if not args.dry_run:
        print(f"\nDataset oppdatert: {CLASSIFIER_DIR}")
    else:
        print("\nKjør uten --dry-run for å faktisk skrive filene.")


if __name__ == "__main__":
    main()
