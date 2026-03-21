"""
Detect and remove outlier annotations using perceptual hashing.

For each category, computes the centroid hash and flags annotations whose
crop looks visually very different from the rest of the category. These are
likely mislabeled (wrong category_id).

Safety rules:
  - Never remove more than 30% of a category's annotations
  - Higher threshold for categories with few annotations (less reliable centroid)
  - Skip categories with < 3 annotations (centroid unreliable)
  - Never touch cat#355 (unknown_product — handled separately)

Outputs:
  ~/Downloads/train/annotations_clean.json  — cleaned COCO file for training
  data/audit_out/removed_annotations.csv    — full list of what was removed + why
  data/audit_out/clean_report.json          — machine-readable summary

Usage:
  python3 data/clean_annotations.py
  python3 data/clean_annotations.py --dry-run
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ANNOTATIONS_PATH    = Path.home() / "Downloads" / "train" / "annotations.json"
ANNOTATIONS_OUT     = Path.home() / "Downloads" / "train" / "annotations_clean.json"
HASH_CACHE_PATH     = Path(__file__).parent / "audit_out" / "hash_cache.json"
OUT_DIR             = Path(__file__).parent / "audit_out"
CLASSIFIER_DIR      = Path(__file__).parent.parent / "datasets" / "classifier"

HASH_SIZE           = 16
THRESHOLD_HIGH      = 0.45   # for categories with ≥ 20 annotations
THRESHOLD_LOW       = 0.42   # for categories with < 20 annotations (stricter)
MAX_REMOVAL_RATIO   = 0.30   # never remove more than 30% of a category
UNKNOWN_CAT_ID      = 355
MIN_ANNS_FOR_CLEAN  = 3      # skip categories with fewer annotations


def hamming(h1, h2):
    return bin(h1 ^ h2).count("1") / (HASH_SIZE * HASH_SIZE)


def centroid_hash(hashes):
    n, total_bits = len(hashes), HASH_SIZE * HASH_SIZE
    counts = [0] * total_bits
    for h in hashes:
        for i in range(total_bits):
            if (h >> (total_bits - 1 - i)) & 1:
                counts[i] += 1
    val = 0
    for c in counts:
        val = (val << 1) | (1 if c > n / 2 else 0)
    return val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without writing files")
    args = parser.parse_args()

    print("Laster data …")
    data = json.load(open(ANNOTATIONS_PATH))
    raw_cache = json.load(open(HASH_CACHE_PATH))
    hash_cache = {}
    for k, v in raw_cache.items():
        if isinstance(v, int):
            hash_cache[int(k)] = v
        elif isinstance(v, str) and v:
            hash_cache[int(k)] = int(v)
        else:
            hash_cache[int(k)] = None

    cats = {c["id"]: c["name"] for c in data["categories"]}
    ann_by_cat = defaultdict(list)
    for a in data["annotations"]:
        ann_by_cat[a["category_id"]].append(a)

    # ── Find outliers ──────────────────────────────────────────────────────────
    removed_ids = set()
    removal_log = []   # list of dicts for CSV

    cats_processed = 0
    cats_cleaned   = 0

    for cid, anns in sorted(ann_by_cat.items()):
        if cid == UNKNOWN_CAT_ID:
            continue
        if len(anns) < MIN_ANNS_FOR_CLEAN:
            continue

        valid = [(a, hash_cache[a["id"]]) for a in anns
                 if hash_cache.get(a["id"]) is not None]
        if len(valid) < MIN_ANNS_FOR_CLEAN:
            continue

        cats_processed += 1
        centroid = centroid_hash([h for _, h in valid])

        # Adaptive threshold
        threshold = THRESHOLD_LOW if len(anns) < 20 else THRESHOLD_HIGH

        scored = sorted([(a, hamming(h, centroid)) for a, h in valid],
                        key=lambda x: -x[1])

        # Apply safety cap: never remove more than MAX_REMOVAL_RATIO
        max_remove = max(1, int(len(anns) * MAX_REMOVAL_RATIO))
        outliers = [(a, d) for a, d in scored if d > threshold][:max_remove]

        if not outliers:
            continue

        cats_cleaned += 1
        for ann, dist in outliers:
            removed_ids.add(ann["id"])
            removal_log.append({
                "ann_id":     ann["id"],
                "image_id":   ann["image_id"],
                "category_id": cid,
                "category_name": cats.get(cid, ""),
                "hamming_dist": round(dist, 4),
                "reason":     f"Visual outlier: Hamming dist {dist:.3f} > threshold {threshold:.2f}",
                "bbox":       ann["bbox"],
            })

    print(f"\nKategorier analysert:  {cats_processed}")
    print(f"Kategorier med feil:   {cats_cleaned}")
    print(f"Annotations fjernet:   {len(removed_ids)}")
    print(f"Annotations beholdt:   {len(data['annotations']) - len(removed_ids)}")
    pct = len(removed_ids) / len(data['annotations']) * 100
    print(f"Andel fjernet:         {pct:.1f}%")

    if args.dry_run:
        print("\n[DRY RUN] Ingen filer skrevet.")
        return removed_ids, removal_log

    # ── Write cleaned annotations.json ────────────────────────────────────────
    clean_anns = [a for a in data["annotations"] if a["id"] not in removed_ids]
    clean_data = dict(data)
    clean_data["annotations"] = clean_anns
    ANNOTATIONS_OUT.write_text(json.dumps(clean_data, ensure_ascii=False), encoding="utf-8")
    print(f"\nClean annotations → {ANNOTATIONS_OUT}")

    # ── Write removed_annotations.csv ─────────────────────────────────────────
    csv_path = OUT_DIR / "removed_annotations.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ann_id", "image_id", "category_id", "category_name",
            "hamming_dist", "reason", "bbox"
        ])
        writer.writeheader()
        writer.writerows(removal_log)
    print(f"Rapport               → {csv_path}  ({len(removal_log)} rader)")

    # ── Write clean_report.json ───────────────────────────────────────────────
    report = {
        "total_original":  len(data["annotations"]),
        "total_removed":   len(removed_ids),
        "total_clean":     len(clean_anns),
        "pct_removed":     round(pct, 2),
        "threshold_high":  THRESHOLD_HIGH,
        "threshold_low":   THRESHOLD_LOW,
        "removed_ids":     sorted(removed_ids),
    }
    (OUT_DIR / "clean_report.json").write_text(json.dumps(report, indent=2))

    # ── Clean classifier dataset ───────────────────────────────────────────────
    print("\nRydder classifier-datasett …")
    removed_from_clf = 0
    for cat_folder in sorted((CLASSIFIER_DIR / "train").iterdir()):
        if not cat_folder.is_dir():
            continue
        for crop_file in list(cat_folder.glob("crop_img*.jpg")):
            # filename: crop_imgXXXXX_annYYYYYY[_jitZ].jpg
            parts = crop_file.stem.split("_")
            try:
                ann_part = [p for p in parts if p.startswith("ann")][0]
                ann_id = int(ann_part[3:])
            except (IndexError, ValueError):
                continue
            if ann_id in removed_ids:
                crop_file.unlink()
                removed_from_clf += 1
        # Also remove aug_ files for removed annotations
        for aug_file in list(cat_folder.glob("aug_crop_img*.jpg")):
            parts = aug_file.stem.split("_")
            try:
                ann_part = [p for p in parts if p.startswith("ann")][0]
                ann_id = int(ann_part[3:])
            except (IndexError, ValueError):
                continue
            if ann_id in removed_ids:
                aug_file.unlink()
                removed_from_clf += 1

    print(f"Fjernet fra classifier: {removed_from_clf} filer")
    print(f"\nFerdig. Din teammate bruker:")
    print(f"  {ANNOTATIONS_OUT}  (som annotations.json for trening)")
    print(f"  {csv_path}          (for å se hva som ble fjernet)")

    return removed_ids, removal_log


if __name__ == "__main__":
    main()
