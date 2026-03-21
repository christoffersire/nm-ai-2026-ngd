"""
Heavy augmentation for low-shot categories using existing GT crops.

For categories with fewer than --min-shots GT crops, generates additional
augmented variants from the existing correct GT crops. Safe because GT crops
are guaranteed to show the right product — no EAN mapping needed.

Augmentations per source crop:
  - Horizontal flip
  - Rotation ±10°, ±20°
  - Brightness jitter (0.6×, 0.8×, 1.2×, 1.5×)
  - Contrast jitter (0.7×, 1.3×)
  - Saturation jitter (0.7×, 1.4×)
  - Random crop 80–95% + resize
  - Combinations: flip+brightness, flip+crop

Usage:
  python3 train/augment_lowshot.py               # threshold ≤10
  python3 train/augment_lowshot.py --min-shots 20
  python3 train/augment_lowshot.py --dry-run
"""

import argparse
import random
from pathlib import Path
from collections import defaultdict, Counter

try:
    from PIL import Image, ImageEnhance, ImageOps
except ImportError:
    raise SystemExit("pip install Pillow")

CLASSIFIER_DIR = Path(__file__).parent.parent / "datasets" / "classifier"
SEED = 42


def augment_variants(img: Image.Image, crop_size: int) -> list[Image.Image]:
    """Return augmented variants of img. All resized to crop_size."""
    base = img.convert("RGB").resize((crop_size, crop_size), Image.LANCZOS)
    variants = []

    def add(i):
        variants.append(i.resize((crop_size, crop_size), Image.LANCZOS))

    flip = ImageOps.mirror(base)

    # Rotations
    for angle in (-20, -10, 10, 20):
        add(base.rotate(angle, expand=False, fillcolor=(128, 128, 128)))

    # Brightness
    for f in (0.6, 0.8, 1.2, 1.5):
        add(ImageEnhance.Brightness(base).enhance(f))
        add(ImageEnhance.Brightness(flip).enhance(f))

    # Contrast
    for f in (0.7, 1.3):
        add(ImageEnhance.Contrast(base).enhance(f))

    # Saturation
    for f in (0.7, 1.4):
        add(ImageEnhance.Color(base).enhance(f))

    # Random crops
    w, h = base.size
    rng = random.Random(SEED)
    for _ in range(4):
        scale = rng.uniform(0.78, 0.95)
        cw, ch = int(w * scale), int(h * scale)
        x0 = rng.randint(0, w - cw)
        y0 = rng.randint(0, h - ch)
        add(base.crop((x0, y0, x0 + cw, y0 + ch)))
        add(flip.crop((x0, y0, x0 + cw, y0 + ch)))

    # Flip alone
    add(flip)

    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-shots", type=int, default=10,
                        help="Augment categories with fewer than this many GT crops (default 10)")
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)
    train_dir = CLASSIFIER_DIR / "train"
    if not train_dir.exists():
        raise SystemExit(f"Ikke funnet: {train_dir}\nKjør prepare_classifier_data.py først.")

    total_written = 0
    total_cats    = 0

    for cat_folder in sorted(train_dir.iterdir()):
        if not cat_folder.is_dir():
            continue

        # Only original GT crops (not jitter, not previously injected)
        gt_crops = [f for f in cat_folder.glob("crop_img*.jpg")
                    if "_jit" not in f.name]
        aug_existing = list(cat_folder.glob("aug_*.jpg"))

        n_gt = len(gt_crops)
        if n_gt == 0 or n_gt >= args.min_shots:
            # Clean up old augmentations if category no longer needs them
            for f in aug_existing:
                if not args.dry_run:
                    f.unlink()
            continue

        cat_id = int(cat_folder.name)

        # Remove stale augmentations before regenerating
        for f in aug_existing:
            if not args.dry_run:
                f.unlink()

        written = 0
        for gt_path in gt_crops:
            try:
                img = Image.open(gt_path)
            except Exception as e:
                print(f"  [WARN] {gt_path.name}: {e}")
                continue

            variants = augment_variants(img, args.crop_size)
            for v_idx, variant in enumerate(variants):
                out_name = f"aug_{gt_path.stem}_v{v_idx:02d}.jpg"
                if not args.dry_run:
                    variant.save(cat_folder / out_name, quality=92)
                written += 1

        total_written += written
        total_cats    += 1
        suffix = " [DRY RUN]" if args.dry_run else ""
        print(f"  cat#{cat_id:3d}  GT={n_gt}  +{written} augmenterte crops{suffix}")

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}=== Ferdig ===")
    print(f"  Kategorier augmentert: {total_cats}")
    print(f"  Bilder skrevet:        {total_written}")
    if not args.dry_run:
        print(f"\n  Dataset: {CLASSIFIER_DIR}")


if __name__ == "__main__":
    main()
