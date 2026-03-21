"""
Product Correction Script
--------------------------
Reads off_cache.json (cached Kassal + OFF responses) and category_mapping.json,
then produces:

  1. data/audit_out/product_corrections.json  — proposed changes, review before applying
  2. data/audit_out/product_corrections.csv   — same, spreadsheet-friendly
  3. (with --apply) updates category_mapping.json in-place with confirmed corrections

Each proposed correction includes:
  - change_type: name_update | code_confirmed | empty_name_filled | discontinued | no_change
  - confidence: high | medium | low
  - our_name / our_code  (current values)
  - kassal_name / off_name (what external sources say)
  - kassal_stores (confirms product is still active)
  - kassal_price
  - recommendation (human-readable)

Usage:
  python3 data/update_products.py              # dry run, write proposals only
  python3 data/update_products.py --apply      # apply high-confidence corrections
  python3 data/update_products.py --apply --min-confidence medium
"""

import argparse
import csv
import difflib
import json
from datetime import date
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

CATEGORY_MAPPING = Path(__file__).parent / "category_mapping.json"
OFF_CACHE        = Path(__file__).parent / "audit_out" / "off_cache.json"
ANNOTATIONS_PATH = Path.home() / "Downloads" / "train" / "annotations.json"
OUT_DIR          = Path(__file__).parent / "audit_out"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def name_sim(a, b):
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def best_kassal_name(our_name, kassal_names):
    """Return (best_name, similarity) from kassal_names."""
    if not kassal_names:
        return None, 0.0
    ranked = sorted(kassal_names, key=lambda n: name_sim(our_name, n), reverse=True)
    best = ranked[0]
    return best, name_sim(our_name, best)


def confidence_level(sim, kassal_found, our_name_empty):
    if our_name_empty and kassal_found:
        return "high"
    if sim >= 0.90:
        return "high"
    if sim >= 0.75:
        return "medium"
    return "low"


# ─── Core logic ───────────────────────────────────────────────────────────────

def build_corrections(mapping, cache, annotations_data):
    # Annotation count per category from actual annotations file
    from collections import Counter
    ann_counts = Counter(a["category_id"] for a in annotations_data["annotations"])
    # Build cat_id → name lookup
    cats_by_id = {c["id"]: c["name"] for c in annotations_data["categories"]}

    corrections = []

    for cid_str, info in mapping.items():
        cid          = int(cid_str)
        our_name     = info.get("product_name") or ""
        our_code     = info.get("product_code") or ""
        ann_count    = ann_counts.get(cid, 0)
        name_empty   = our_name.strip() == ""

        if not our_code:
            corrections.append({
                "category_id":    cid,
                "our_name":       our_name,
                "our_code":       our_code,
                "change_type":    "no_product_code",
                "confidence":     "low",
                "kassal_name":    "",
                "kassal_names_all": "",
                "off_name":       "",
                "name_similarity": 0.0,
                "kassal_found":   False,
                "kassal_stores":  "",
                "kassal_price":   "",
                "kassal_image_url": "",
                "discontinued":   False,
                "annotation_count": ann_count,
                "recommendation": "No product code — cannot look up externally. Needs manual review.",
                "proposed_name":  "",
                "proposed_code":  our_code,
            })
            continue

        kassal_key = f"kassal_{our_code}"
        kdata      = cache.get(kassal_key, {})
        off_data   = cache.get(our_code, {})

        kassal_found  = kdata.get("found", False)
        kassal_names  = kdata.get("names", [])
        kassal_stores = ", ".join(kdata.get("stores", []))
        kassal_price  = kdata.get("current_price")
        kassal_imgurl = kdata.get("image_url") or ""
        off_name      = off_data.get("product_name") or "" if isinstance(off_data, dict) else ""

        discontinued = not kassal_found

        if name_empty and kassal_found and kassal_names:
            # Fill empty name from Kassal
            proposed_name = kassal_names[0]
            sim = 1.0  # empty → any kassal name is an improvement
            change_type = "empty_name_filled"
            conf = "high"
            rec = f'Fill empty name with Kassal: "{proposed_name}"'

        elif discontinued:
            proposed_name = our_name
            sim = 0.0
            change_type = "discontinued"
            conf = "high"
            rec = f'Not found in Kassal (HTTP 404). Likely discontinued. Consider reclassifying {ann_count} annotations as unknown_product (cat 355).'

        else:
            best_name, sim = best_kassal_name(our_name, kassal_names)
            conf = confidence_level(sim, kassal_found, name_empty)

            if sim >= 0.90:
                proposed_name = our_name  # already good
                change_type = "name_confirmed"
                rec = f'Name confirmed by Kassal (sim={sim:.2f}). No change needed.'
            elif sim >= 0.75:
                proposed_name = best_name
                change_type = "name_update"
                rec = f'Minor name difference (sim={sim:.2f}). Kassal: "{best_name}". Consider updating.'
            else:
                proposed_name = best_name or our_name
                change_type = "name_mismatch"
                rec = f'Significant name mismatch (sim={sim:.2f}). Our: "{our_name}". Kassal: "{best_name}". Manual review recommended.'

        corrections.append({
            "category_id":      cid,
            "our_name":         our_name,
            "our_code":         our_code,
            "change_type":      change_type,
            "confidence":       conf,
            "kassal_name":      kassal_names[0] if kassal_names else "",
            "kassal_names_all": " | ".join(kassal_names),
            "off_name":         off_name,
            "name_similarity":  round(sim, 4),
            "kassal_found":     kassal_found,
            "kassal_stores":    kassal_stores,
            "kassal_price":     kassal_price if kassal_price is not None else "",
            "kassal_image_url": kassal_imgurl,
            "discontinued":     discontinued,
            "annotation_count": ann_count,
            "recommendation":   rec,
            "proposed_name":    proposed_name,
            "proposed_code":    our_code,
        })

    # Sort: actionable first
    order = {
        "empty_name_filled": 0,
        "discontinued":      1,
        "name_mismatch":     2,
        "name_update":       3,
        "no_product_code":   4,
        "name_confirmed":    5,
    }
    corrections.sort(key=lambda x: (order.get(x["change_type"], 9), -x["annotation_count"]))
    return corrections


def apply_corrections(mapping, corrections, min_conf):
    conf_rank = {"high": 2, "medium": 1, "low": 0}
    min_rank  = conf_rank.get(min_conf, 1)
    applied   = 0
    for c in corrections:
        if conf_rank.get(c["confidence"], 0) < min_rank:
            continue
        if c["change_type"] in ("name_confirmed", "no_product_code", "discontinued"):
            continue  # don't auto-apply these
        cid_str = str(c["category_id"])
        if cid_str in mapping:
            old_name = mapping[cid_str].get("product_name", "")
            new_name = c["proposed_name"]
            if new_name and new_name != old_name:
                mapping[cid_str]["product_name"] = new_name
                mapping[cid_str]["_kassal_confirmed_name"] = c["kassal_name"]
                mapping[cid_str]["_kassal_stores"]         = c["kassal_stores"]
                mapping[cid_str]["_kassal_price"]          = c["kassal_price"]
                mapping[cid_str]["_kassal_image_url"]      = c["kassal_image_url"]
                mapping[cid_str]["_correction_date"]       = str(date.today())
                applied += 1
    return applied


# ─── Report writers ───────────────────────────────────────────────────────────

CSV_COLS = [
    "category_id", "change_type", "confidence", "annotation_count",
    "our_name", "our_code",
    "kassal_name", "kassal_names_all", "off_name",
    "name_similarity",
    "kassal_found", "kassal_stores", "kassal_price", "kassal_image_url",
    "discontinued",
    "proposed_name", "proposed_code",
    "recommendation",
]


def write_csv(corrections, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(corrections)


def print_summary(corrections):
    from collections import Counter
    ct = Counter(c["change_type"] for c in corrections)
    cf = Counter(c["confidence"] for c in corrections)
    print("\n" + "=" * 60)
    print("PRODUCT CORRECTION SUMMARY")
    print("=" * 60)
    print(f"{'Change type':<25} {'Count':>6}")
    for k, v in sorted(ct.items()):
        print(f"  {k:<23} {v:>6}")
    print(f"\n{'Confidence':<25} {'Count':>6}")
    for k, v in sorted(cf.items()):
        print(f"  {k:<23} {v:>6}")

    print("\n--- Actionable items ---")
    for c in corrections:
        if c["change_type"] in ("empty_name_filled", "name_mismatch", "name_update", "discontinued"):
            disc = " [DISCONTINUED]" if c["discontinued"] else ""
            print(f"  cat#{c['category_id']} [{c['confidence']}] {c['change_type']}{disc}")
            print(f"    Our name:  {c['our_name'] or '(empty)'}")
            if c["kassal_name"]:
                print(f"    Kassal:    {c['kassal_name']}")
            print(f"    Stores:    {c['kassal_stores'] or '—'}")
            print(f"    Rec: {c['recommendation'][:100]}")
            print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply",          action="store_true", help="Apply corrections to category_mapping.json")
    parser.add_argument("--min-confidence", default="high",      help="Minimum confidence to apply (high/medium/low)")
    args = parser.parse_args()

    print("Loading data …")
    mapping  = json.load(open(CATEGORY_MAPPING))
    cache    = json.load(open(OFF_CACHE))
    ann_data = json.load(open(ANNOTATIONS_PATH))

    corrections = build_corrections(mapping, cache, ann_data)

    # Write outputs
    OUT_DIR.mkdir(exist_ok=True)
    json_path = OUT_DIR / "product_corrections.json"
    csv_path  = OUT_DIR / "product_corrections.csv"

    json.dump(corrections, open(json_path, "w"), indent=2, ensure_ascii=False)
    write_csv(corrections, csv_path)
    print(f"Proposals written → {json_path}")
    print(f"Proposals written → {csv_path}")

    print_summary(corrections)

    if args.apply:
        n = apply_corrections(mapping, corrections, args.min_confidence)
        CATEGORY_MAPPING.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
        print(f"\nApplied {n} corrections to {CATEGORY_MAPPING}")
    else:
        print("\nDry run — no changes written.")
        print("Run with --apply to apply high-confidence corrections.")
        print("Run with --apply --min-confidence medium to also apply medium-confidence ones.")


if __name__ == "__main__":
    main()
