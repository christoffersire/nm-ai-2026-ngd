"""
Embedding-based annotation cleaning pipeline.

Uses a pretrained ConvNeXt to extract embeddings for:
  1. All product reference images (ground truth)
  2. All annotation crops from training images

Then matches each crop to the nearest reference product by cosine similarity.
Where the nearest product differs from the assigned category_id, flags it as mislabeled.

Usage:
  python data/scripts/clean_embeddings.py                          # Dry run
  python data/scripts/clean_embeddings.py --apply                  # Write corrected annotations
  python data/scripts/clean_embeddings.py --apply --min-sim 0.3    # Lower similarity threshold
"""
import argparse
import json
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm


# --- Paths ---
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

ANNOTATIONS_PATH = RAW_DIR / "annotations.json"
METADATA_PATH = DATA_DIR / "metadata.json"
IMAGE_DIR = RAW_DIR / "images"
REF_IMAGE_DIR = RAW_DIR / "product_images"
OUTPUT_PATH = RAW_DIR / "annotations_cleaned.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EMBED_MODEL = "convnext_base.fb_in22k"
EMBED_DIM = 1024
IMG_SIZE = 224


def build_model():
    model = timm.create_model(EMBED_MODEL, pretrained=True, num_classes=0)
    model = model.to(DEVICE).eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def extract_embeddings_batch(model, images, transform):
    """Extract embeddings for a list of PIL images."""
    tensors = torch.stack([transform(img.convert("RGB")) for img in images])
    tensors = tensors.to(DEVICE)
    embeddings = model(tensors)
    embeddings = F.normalize(embeddings, dim=1)
    return embeddings.cpu().numpy()


def build_reference_embeddings(model, transform, cat_id_to_code):
    """Build average embedding per category from reference product images."""
    ref_embeddings = {}
    ref_views = ["main.jpg", "front.jpg", "back.jpg", "left.jpg", "right.jpg"]

    print(f"Building reference embeddings for {len(cat_id_to_code)} categories...")
    for cat_id, code in cat_id_to_code.items():
        product_dir = REF_IMAGE_DIR / code
        if not product_dir.exists():
            continue

        images = []
        for view in ref_views:
            view_path = product_dir / view
            if view_path.exists():
                try:
                    images.append(Image.open(view_path))
                except Exception:
                    continue

        if not images:
            continue

        embeddings = extract_embeddings_batch(model, images, transform)
        ref_embeddings[cat_id] = embeddings.mean(axis=0)
        ref_embeddings[cat_id] /= np.linalg.norm(ref_embeddings[cat_id])

    print(f"  Built embeddings for {len(ref_embeddings)} categories")
    return ref_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write corrected annotations")
    parser.add_argument("--min-sim", type=float, default=0.25,
                        help="Minimum cosine similarity to accept a correction")
    parser.add_argument("--margin", type=float, default=0.05,
                        help="Required margin: best_sim - gt_sim to trigger correction")
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

    # Build category -> product_code mapping
    cat_id_to_code = {}
    corrected_by_cat = {}
    for p in meta["products"]:
        if p["product_name"] in cat_name_to_id:
            cid = cat_name_to_id[p["product_name"]]
            cat_id_to_code[cid] = p["product_code"]
            corrected_by_cat[cid] = p.get("corrected_count", 0)

    # Build model and reference embeddings
    print(f"Loading {EMBED_MODEL} on {DEVICE}...")
    model = build_model()
    transform = get_transform()
    ref_embeddings = build_reference_embeddings(model, transform, cat_id_to_code)

    # Build reference embedding matrix for fast comparison
    ref_cat_ids = sorted(ref_embeddings.keys())
    ref_matrix = np.stack([ref_embeddings[cid] for cid in ref_cat_ids])  # (N_ref, D)
    ref_cat_id_lookup = {i: cid for i, cid in enumerate(ref_cat_ids)}

    print(f"Reference matrix: {ref_matrix.shape}")

    # Group annotations by image
    ann_by_img = defaultdict(list)
    for a in data["annotations"]:
        ann_by_img[a["image_id"]].append(a)

    # Process all annotations
    print(f"\nProcessing {len(data['annotations'])} annotations across {len(ann_by_img)} images...")
    t0 = time.time()

    corrections = {}  # ann_id -> (new_cat_id, gt_sim, best_sim, best_cat_id)
    stats = {"total": 0, "has_ref": 0, "mismatch": 0, "corrected": 0}

    image_ids = sorted(ann_by_img.keys())
    for idx, img_id in enumerate(image_ids):
        img_info = id_to_img[img_id]
        img_path = IMAGE_DIR / img_info["file_name"]
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        anns = ann_by_img[img_id]

        # Crop all annotations from this image
        crops = []
        valid_anns = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            # Clamp to image bounds
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(img_w, int(x + w))
            y2 = min(img_h, int(y + h))
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            crop = img.crop((x1, y1, x2, y2))
            crops.append(crop)
            valid_anns.append(ann)

        if not crops:
            continue

        # Extract embeddings in batches
        all_crop_embeds = []
        for i in range(0, len(crops), BATCH_SIZE):
            batch = crops[i:i + BATCH_SIZE]
            embeds = extract_embeddings_batch(model, batch, transform)
            all_crop_embeds.append(embeds)
        crop_embeddings = np.concatenate(all_crop_embeds, axis=0)  # (N_crops, D)

        # Compare each crop to all reference products
        similarities = crop_embeddings @ ref_matrix.T  # (N_crops, N_ref)

        for j, ann in enumerate(valid_anns):
            stats["total"] += 1
            gt_cat = ann["category_id"]

            # Find best matching reference product
            best_ref_idx = similarities[j].argmax()
            best_cat_id = ref_cat_id_lookup[best_ref_idx]
            best_sim = float(similarities[j, best_ref_idx])

            # Get similarity to GT category's reference (if it has one)
            gt_sim = -1.0
            if gt_cat in ref_embeddings:
                stats["has_ref"] += 1
                gt_ref_idx = ref_cat_ids.index(gt_cat)
                gt_sim = float(similarities[j, gt_ref_idx])

            # Decide on correction
            if best_cat_id != gt_cat:
                stats["mismatch"] += 1

                # Correction criteria:
                # 1. Best match must be above minimum similarity
                # 2. Best match must beat GT match by margin
                # 3. GT category must have known corruption (corrected_count > 0)
                #    OR GT category is not in metadata at all
                gt_has_corruption = corrected_by_cat.get(gt_cat, -1) != 0  # -1 = not in metadata
                margin_ok = (gt_sim < 0) or (best_sim - gt_sim >= args.margin)

                if best_sim >= args.min_sim and margin_ok and gt_has_corruption:
                    corrections[ann["id"]] = {
                        "ann_id": ann["id"],
                        "image_id": ann["image_id"],
                        "gt_cat": gt_cat,
                        "gt_name": cat_id_to_name.get(gt_cat, "?"),
                        "new_cat": best_cat_id,
                        "new_name": cat_id_to_name.get(best_cat_id, "?"),
                        "gt_sim": round(gt_sim, 4),
                        "best_sim": round(best_sim, 4),
                        "margin": round(best_sim - gt_sim, 4) if gt_sim >= 0 else None,
                    }
                    stats["corrected"] += 1

        if (idx + 1) % 25 == 0 or idx == len(image_ids) - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(image_ids) - idx - 1) / rate
            print(f"  [{idx+1}/{len(image_ids)}] {rate:.1f} img/s, "
                  f"corrections so far: {stats['corrected']}, ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Total annotations: {stats['total']}")
    print(f"  With reference: {stats['has_ref']}")
    print(f"  Embedding mismatch: {stats['mismatch']}")
    print(f"  Corrections proposed: {stats['corrected']}")

    # Analyze corrections by category
    corr_list = sorted(corrections.values(), key=lambda x: -(x["best_sim"]))

    changes_from = Counter(c["gt_cat"] for c in corr_list)
    changes_to = Counter(c["new_cat"] for c in corr_list)

    print(f"\nTop 20 categories losing annotations:")
    print(f"{'Cat':>4} {'#Lost':>6} {'CorrCount':>9}  Name")
    print("-" * 80)
    for cat_id, count in changes_from.most_common(20):
        corr = corrected_by_cat.get(cat_id, "N/A")
        name = cat_id_to_name.get(cat_id, "?")
        print(f"{cat_id:>4} {count:>6} {str(corr):>9}  {name[:55]}")

    print(f"\nTop 20 categories gaining annotations:")
    print(f"{'Cat':>4} {'#Gained':>7}  Name")
    print("-" * 80)
    for cat_id, count in changes_to.most_common(20):
        name = cat_id_to_name.get(cat_id, "?")
        print(f"{cat_id:>4} {count:>7}  {name[:55]}")

    print(f"\nTop 40 highest-similarity corrections:")
    print(f"{'BestSim':>7} {'GTSim':>6} {'Margin':>6} {'GT':>4} -> {'New':>4}  GT_Name -> New_Name")
    print("-" * 110)
    for c in corr_list[:40]:
        gt_n = c["gt_name"][:28]
        new_n = c["new_name"][:28]
        margin = f"{c['margin']:+.3f}" if c["margin"] is not None else "  N/A"
        print(f"{c['best_sim']:7.4f} {c['gt_sim']:6.3f} {margin:>6} {c['gt_cat']:>4} -> {c['new_cat']:>4}  {gt_n} -> {new_n}")

    # Save corrections
    corr_path = DATA_DIR / "embedding_corrections.json"
    with open(corr_path, "w") as f:
        json.dump(corr_list, f, indent=2, ensure_ascii=False)
    print(f"\nAll {len(corr_list)} corrections saved to {corr_path}")

    # Apply corrections
    if args.apply:
        print(f"\n{'='*60}")
        print("APPLYING CORRECTIONS")
        print(f"{'='*60}")

        corrected_data = json.loads(json.dumps(data))  # deep copy
        n_changed = 0
        for ann in corrected_data["annotations"]:
            if ann["id"] in corrections:
                ann["category_id"] = corrections[ann["id"]]["new_cat"]
                n_changed += 1

        print(f"Changed {n_changed} / {len(data['annotations'])} annotations")

        # Show count shifts for most-affected categories
        old_counts = Counter(a["category_id"] for a in data["annotations"])
        new_counts = Counter(a["category_id"] for a in corrected_data["annotations"])
        all_cats = set(old_counts.keys()) | set(new_counts.keys())
        shifts = [(cid, old_counts.get(cid, 0), new_counts.get(cid, 0)) for cid in all_cats]
        shifts = [(cid, o, n, n - o) for cid, o, n in shifts if o != n]
        shifts.sort(key=lambda x: abs(x[3]), reverse=True)

        print(f"\nTop 20 category count changes:")
        for cid, old, new, diff in shifts[:20]:
            name = cat_id_to_name.get(cid, "?")[:40]
            corr = corrected_by_cat.get(cid, "?")
            print(f"  cat {cid:>3}: {old:>4} -> {new:>4} ({diff:>+4})  corr_count={corr}  {name}")

        with open(OUTPUT_PATH, "w") as f:
            json.dump(corrected_data, f)
        print(f"\nWritten to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
