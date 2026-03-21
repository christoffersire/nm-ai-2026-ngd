"""
Verify all product data against Kassal.app API (authoritative Norwegian grocery database).

For every category in our competition dataset:
1. Search Kassal.app by product name
2. Verify EAN/GTIN matches our metadata
3. Download reference images for categories missing them
4. Flag any name/EAN discrepancies

Usage:
  python data/scripts/verify_products.py
  python data/scripts/verify_products.py --download-missing   # Also download missing images
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REF_DIR = RAW_DIR / "product_images"

API_BASE = "https://kassal.app/api/v1"
API_KEY = "hLW4tjE0ZxddvLsk9Ow2uNPVUDXV58aLGojGFjfm"

ANNOTATIONS_PATH = RAW_DIR / "annotations.json"
METADATA_PATH = DATA_DIR / "metadata.json"


def api_search(query, size=5):
    """Search Kassal.app products API."""
    url = f"{API_BASE}/products?search={urllib.parse.quote(query)}&size={size}&exclude_without_ean=1"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def api_ean_lookup(ean):
    """Look up a product by EAN."""
    url = f"{API_BASE}/products/ean/{ean}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def download_image(url, output_path):
    """Download image from URL."""
    try:
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=15)
        data = resp.read()
        if len(data) < 500:
            return False, f"too small ({len(data)}B)"
        output_path.write_bytes(data)
        return True, f"{len(data) // 1024}KB"
    except Exception as e:
        return False, str(e)


import urllib.parse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-missing", action="store_true",
                        help="Download reference images for categories that don't have them")
    args = parser.parse_args()

    # Load data
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)
    with open(METADATA_PATH) as f:
        meta = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    cat_name_to_id = {c["name"]: c["id"] for c in data["categories"]}

    # Build metadata lookup
    meta_by_name = {}
    for p in meta["products"]:
        meta_by_name[p["product_name"]] = p

    # Check which categories have reference images
    def has_ref_image(cat_id):
        if cat_id_to_name[cat_id] in meta_by_name:
            code = meta_by_name[cat_id_to_name[cat_id]]["product_code"]
            product_dir = REF_DIR / code
            if product_dir.exists() and any(product_dir.iterdir()):
                return True
        return False

    # Process all categories
    results = []
    missing_images = []
    discrepancies = []

    categories = sorted(data["categories"], key=lambda c: c["id"])
    total = len(categories)

    print(f"Verifying {total} categories against Kassal.app API...")
    print(f"{'='*90}")

    for i, cat in enumerate(categories):
        cat_id = cat["id"]
        cat_name = cat["name"]
        has_ref = has_ref_image(cat_id)
        meta_info = meta_by_name.get(cat_name)
        meta_ean = meta_info["product_code"] if meta_info else None

        # Skip unknown_product
        if cat_name == "unknown_product":
            results.append({
                "cat_id": cat_id,
                "cat_name": cat_name,
                "status": "SKIP",
                "reason": "unknown_product category",
                "has_ref_image": has_ref,
            })
            continue

        # Rate limit
        if i > 0 and i % 10 == 0:
            time.sleep(1)

        # Search by name
        search_result = api_search(cat_name, size=5)

        if "error" in search_result:
            results.append({
                "cat_id": cat_id,
                "cat_name": cat_name,
                "status": "API_ERROR",
                "error": search_result["error"],
                "has_ref_image": has_ref,
            })
            print(f"  [{i+1}/{total}] cat {cat_id:>3}: API ERROR — {cat_name}")
            continue

        api_products = search_result.get("data", [])

        # Find best match
        best_match = None
        exact_name_match = None
        ean_match = None

        for p in api_products:
            p_name = p.get("name", "").upper()
            p_ean = p.get("ean", "")

            # Check exact name match (case-insensitive)
            if p_name == cat_name.upper() or p_name.replace("  ", " ") == cat_name.upper():
                exact_name_match = p
                break

            # Check EAN match
            if meta_ean and p_ean == meta_ean:
                ean_match = p

        best_match = exact_name_match or ean_match or (api_products[0] if api_products else None)

        # Determine status
        status = "OK"
        issues = []

        if not api_products:
            status = "NOT_FOUND"
            issues.append("No results from Kassal.app")
        elif not exact_name_match:
            if ean_match:
                status = "NAME_MISMATCH"
                issues.append(f"EAN matches but name differs: API='{ean_match['name']}' vs DS='{cat_name}'")
            elif best_match:
                status = "FUZZY_MATCH"
                issues.append(f"Best guess: '{best_match['name']}' (EAN: {best_match.get('ean', 'N/A')})")

        if best_match and meta_ean and best_match.get("ean") and best_match["ean"] != meta_ean:
            status = "EAN_MISMATCH"
            issues.append(f"EAN differs: API={best_match['ean']} vs metadata={meta_ean}")

        if not has_ref:
            issues.append("MISSING reference image")
            if best_match and best_match.get("image"):
                missing_images.append({
                    "cat_id": cat_id,
                    "cat_name": cat_name,
                    "ean": best_match.get("ean"),
                    "image_url": best_match["image"],
                    "api_name": best_match["name"],
                })

        result = {
            "cat_id": cat_id,
            "cat_name": cat_name,
            "status": status,
            "has_ref_image": has_ref,
            "meta_ean": meta_ean,
            "api_name": best_match["name"] if best_match else None,
            "api_ean": best_match.get("ean") if best_match else None,
            "api_image": best_match.get("image") if best_match else None,
            "api_brand": best_match.get("brand") if best_match else None,
            "issues": issues,
        }
        results.append(result)

        if issues:
            discrepancies.append(result)

        # Status indicator
        icon = {"OK": "✓", "NOT_FOUND": "✗", "NAME_MISMATCH": "~",
                "FUZZY_MATCH": "?", "EAN_MISMATCH": "!", "API_ERROR": "E",
                "SKIP": "-"}.get(status, "?")
        extra = f" — {'; '.join(issues)}" if issues else ""
        print(f"  [{i+1}/{total}] {icon} cat {cat_id:>3}: {status:15s} {cat_name[:45]}{extra}")

    # Summary
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")

    from collections import Counter
    status_counts = Counter(r["status"] for r in results)
    for status, count in status_counts.most_common():
        print(f"  {status:20s}: {count}")

    print(f"\n  Missing reference images: {len(missing_images)}")
    print(f"  Total discrepancies: {len(discrepancies)}")

    # Save full results
    output_path = DATA_DIR / "product_verification.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": dict(status_counts),
            "results": results,
            "missing_images": missing_images,
            "discrepancies": discrepancies,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results saved to: {output_path}")

    # Download missing images
    if args.download_missing and missing_images:
        print(f"\n{'='*90}")
        print(f"DOWNLOADING {len(missing_images)} MISSING REFERENCE IMAGES")
        print(f"{'='*90}")

        for item in missing_images:
            ean = item["ean"] or f"cat{item['cat_id']}"
            out_dir = REF_DIR / ean
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "main.jpg"

            if out_path.exists():
                print(f"  cat {item['cat_id']:>3}: already exists — {item['cat_name']}")
                continue

            success, msg = download_image(item["image_url"], out_path)
            icon = "✓" if success else "✗"
            print(f"  cat {item['cat_id']:>3}: {icon} {msg} — {item['cat_name']} (EAN: {ean})")
            time.sleep(0.5)


if __name__ == "__main__":
    main()
