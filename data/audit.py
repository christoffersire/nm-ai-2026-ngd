"""
Quick data audit: class frequencies, bbox stats, mapping table, duplicate check.
"""
import json
from pathlib import Path
from collections import Counter


ANNOTATIONS_PATH = Path.home() / "Downloads" / "train" / "annotations.json"
PRODUCT_META_PATH = Path.home() / "Downloads" / "NM_NGD_product_images" / "metadata.json"
PRODUCT_IMAGES_DIR = Path.home() / "Downloads" / "NM_NGD_product_images"


def main():
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)

    with open(PRODUCT_META_PATH) as f:
        meta = json.load(f)

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    print("=" * 60)
    print("DATA AUDIT")
    print("=" * 60)

    # Basic counts
    print(f"\nImages: {len(images)}")
    print(f"Annotations: {len(annotations)}")
    print(f"Categories: {len(categories)}")
    cat_ids = sorted(c["id"] for c in categories)
    print(f"Category ID range: {cat_ids[0]}-{cat_ids[-1]}")
    print(f"Sequential: {cat_ids == list(range(cat_ids[0], cat_ids[-1]+1))}")

    # Image resolution stats
    print(f"\n--- Image Resolutions ---")
    widths = [img["width"] for img in images]
    heights = [img["height"] for img in images]
    areas = [w * h for w, h in zip(widths, heights)]
    print(f"Width: min={min(widths)}, max={max(widths)}, mean={sum(widths)/len(widths):.0f}")
    print(f"Height: min={min(heights)}, max={max(heights)}, mean={sum(heights)/len(heights):.0f}")
    print(f"Megapixels: min={min(areas)/1e6:.1f}, max={max(areas)/1e6:.1f}, mean={sum(areas)/len(areas)/1e6:.1f}")

    # Annotations per image
    print(f"\n--- Annotations per Image ---")
    ann_per_img = Counter(a["image_id"] for a in annotations)
    counts = list(ann_per_img.values())
    print(f"Min: {min(counts)}, Max: {max(counts)}, Mean: {sum(counts)/len(counts):.1f}")

    # Category frequency distribution
    print(f"\n--- Category Frequency Distribution ---")
    cat_counts = Counter(a["category_id"] for a in annotations)
    freq_buckets = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 200), (201, 500), (501, 1000)]
    for lo, hi in freq_buckets:
        n = sum(1 for c in cat_counts.values() if lo <= c <= hi)
        if n > 0:
            print(f"  {lo:>4}-{hi:>4} annotations: {n} categories")

    # Single-annotation categories
    single_cats = [cid for cid, cnt in cat_counts.items() if cnt == 1]
    print(f"\nSingle-annotation categories: {len(single_cats)}")

    # Bbox area distribution
    print(f"\n--- Bbox Area Distribution ---")
    areas = [a["area"] for a in annotations]
    print(f"Min: {min(areas):.0f}, Max: {max(areas):.0f}, Mean: {sum(areas)/len(areas):.0f}")
    print(f"Median: {sorted(areas)[len(areas)//2]:.0f}")

    # Small boxes (< 32x32 = 1024 px^2)
    small = sum(1 for a in areas if a < 1024)
    print(f"Tiny boxes (<32x32): {small} ({100*small/len(areas):.1f}%)")

    # Mapping table: category → product reference images
    print(f"\n--- Reference Image Mapping ---")
    cat_name_to_id = {c["name"]: c["id"] for c in categories}

    matched = 0
    unmatched = []
    mapping = {}
    for p in meta["products"]:
        pname = p["product_name"]
        pcode = p["product_code"]
        if pname in cat_name_to_id:
            cat_id = cat_name_to_id[pname]
            mapping[cat_id] = {
                "product_code": pcode,
                "product_name": pname,
                "has_images": p["has_images"],
                "image_types": p["image_types"],
                "annotation_count": p["annotation_count"],
            }
            matched += 1
        else:
            unmatched.append({"product_code": pcode, "product_name": pname})

    print(f"Matched by exact name: {matched}/{len(meta['products'])}")
    if unmatched:
        print(f"Unmatched products:")
        for u in unmatched:
            print(f"  {u['product_code']}: {u['product_name']}")

    # Categories without reference images
    cats_with_ref = set(mapping.keys())
    cats_without_ref = [c["id"] for c in categories if c["id"] not in cats_with_ref]
    print(f"Categories without reference images: {len(cats_without_ref)}")
    for cid in cats_without_ref[:10]:
        cat_name = next(c["name"] for c in categories if c["id"] == cid)
        cnt = cat_counts.get(cid, 0)
        print(f"  id={cid}: {cat_name} ({cnt} annotations)")

    # Check for missing image types
    missing_views = []
    for cat_id, info in mapping.items():
        if not info["has_images"]:
            missing_views.append((cat_id, info["product_name"]))
    if missing_views:
        print(f"\nProducts with folders but no images: {len(missing_views)}")
        for cid, name in missing_views:
            print(f"  id={cid}: {name}")

    # Duplicate check: simple filename pattern analysis
    print(f"\n--- Duplicate/Near-Duplicate Check ---")
    fnames = [img["file_name"] for img in images]
    # Check for sequential gaps (might indicate related captures)
    ids_sorted = sorted(img["id"] for img in images)
    gaps = []
    for i in range(1, len(ids_sorted)):
        if ids_sorted[i] - ids_sorted[i-1] > 1:
            gaps.append((ids_sorted[i-1], ids_sorted[i]))
    print(f"Image ID gaps: {len(gaps)} (IDs are not fully sequential)")
    print(f"ID range: {ids_sorted[0]}-{ids_sorted[-1]}")

    # Save mapping for later use
    mapping_path = Path.home() / "nm-ai-2026-ngd" / "data" / "category_mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
    print(f"\nCategory mapping saved to {mapping_path}")


if __name__ == "__main__":
    main()
