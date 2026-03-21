"""
Apply annotation patches from annotation_patches.json to annotations.json.

Creates a corrected copy at ~/Downloads/train/annotations_patched.json
(never overwrites the original).

Usage:
  python3 data/apply_patches.py          # dry run — show what would change
  python3 data/apply_patches.py --apply  # write annotations_patched.json
"""

import argparse
import json
from pathlib import Path

PATCHES_FILE   = Path(__file__).parent / "annotation_patches.json"
ANNOTATIONS_IN = Path.home() / "Downloads" / "train" / "annotations.json"
ANNOTATIONS_OUT = Path.home() / "Downloads" / "train" / "annotations_patched.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write patched file")
    args = parser.parse_args()

    patches_data = json.load(open(PATCHES_FILE))
    patches = {p["annotation_id"]: p for p in patches_data["patches"]}

    data = json.load(open(ANNOTATIONS_IN))
    cats = {c["id"]: c["name"] for c in data["categories"]}

    applied = []
    skipped = []

    new_annotations = []
    for ann in data["annotations"]:
        patch = patches.get(ann["id"])
        if patch is None:
            new_annotations.append(ann)
            continue

        # Validate old_category_id matches current state
        if ann["category_id"] != patch["old_category_id"]:
            skipped.append({
                "annotation_id": ann["id"],
                "reason": f"Expected old_category_id={patch['old_category_id']} "
                          f"but found {ann['category_id']} ({cats.get(ann['category_id'], '?')}). "
                          f"Already patched or wrong patch?",
            })
            new_annotations.append(ann)
            continue

        new_ann = dict(ann)
        new_ann["category_id"] = patch["new_category_id"]
        new_annotations.append(new_ann)
        applied.append({
            "annotation_id": ann["id"],
            "image_id": ann["image_id"],
            "old": f"cat#{patch['old_category_id']} ({patch['old_name']})",
            "new": f"cat#{patch['new_category_id']} ({patch['new_name']})",
            "reason": patch["reason"],
        })

    # Summary
    print(f"Patches defined : {len(patches)}")
    print(f"Would apply     : {len(applied)}")
    print(f"Skipped         : {len(skipped)}")
    print()

    for a in applied:
        print(f"  ann#{a['annotation_id']} img#{a['image_id']}")
        print(f"    {a['old']}")
        print(f"    → {a['new']}")
        print(f"    {a['reason']}")
        print()

    if skipped:
        print("SKIPPED (mismatch — check patch file):")
        for s in skipped:
            print(f"  ann#{s['annotation_id']}: {s['reason']}")
        print()

    if args.apply:
        new_data = dict(data)
        new_data["annotations"] = new_annotations
        ANNOTATIONS_OUT.write_text(json.dumps(new_data, ensure_ascii=False))
        print(f"Written → {ANNOTATIONS_OUT}")
        print("Use this file for YOLO dataset preparation instead of annotations.json")
    else:
        print("Dry run — pass --apply to write the patched file.")


if __name__ == "__main__":
    main()
