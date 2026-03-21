"""
Download product images from NorgesGruppen's public CDN (bilder.ngdata.no).

Downloads large product images for all categories in category_mapping.json.
Images are saved as data/product_images/{cat_id}_{ean}.jpg

Usage:
  python data/download_product_images.py
  python data/download_product_images.py --sizes large,medium --stores meny,kiwi
"""
import argparse
import json
import urllib.request
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"

CATEGORY_MAPPING = DATA_DIR / "category_mapping.json"
OUTPUT_DIR = DATA_DIR / "product_images"
CDN_BASE = "https://bilder.ngdata.no"


def download_image(cat_id: str, ean: str, store: str, size: str, output_dir: Path) -> tuple[str, bool, str]:
    """Download a single product image. Returns (filename, success, message)."""
    suffix = f"_{store}_{size}" if store != "meny" or size != "large" else ""
    filename = f"cat{int(cat_id):03d}_{ean}{suffix}.jpg"
    output_path = output_dir / filename

    if output_path.exists() and output_path.stat().st_size > 1000:
        return filename, True, "exists"

    url = f"{CDN_BASE}/{ean}/{store}/{size}.jpg"
    try:
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=15)
        data = resp.read()
        if len(data) < 1000:
            return filename, False, f"too small ({len(data)}B)"
        output_path.write_bytes(data)
        return filename, True, f"{len(data) // 1024}KB"
    except Exception as e:
        return filename, False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="large", help="Comma-separated sizes (large,medium,small)")
    parser.add_argument("--stores", default="meny", help="Comma-separated stores (meny,kiwi)")
    parser.add_argument("--workers", type=int, default=8, help="Download threads")
    args = parser.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]
    stores = [s.strip() for s in args.stores.split(",")]

    with open(CATEGORY_MAPPING) as f:
        cats = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build download tasks
    tasks = []
    for cat_id, info in cats.items():
        ean = info.get("product_code", "")
        if not ean or not ean.isdigit():
            continue
        for store in stores:
            for size in sizes:
                tasks.append((cat_id, ean, store, size))

    print(f"Downloading {len(tasks)} images ({len(cats)} categories × {len(stores)} stores × {len(sizes)} sizes)")
    print(f"Output: {OUTPUT_DIR}")

    success = 0
    fail = 0
    fail_list = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_image, cat_id, ean, store, size, OUTPUT_DIR): (cat_id, ean)
            for cat_id, ean, store, size in tasks
        }

        for i, future in enumerate(as_completed(futures), 1):
            filename, ok, msg = future.result()
            if ok:
                success += 1
            else:
                fail += 1
                fail_list.append(f"{filename}: {msg}")

            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} — {success} ok, {fail} fail")

    print(f"\n=== Done: {success} downloaded, {fail} failed ===")
    if fail_list:
        print(f"\nFailed ({len(fail_list)}):")
        for f in fail_list:
            print(f"  {f}")

    # Write summary
    summary = {
        "total_categories": len(cats),
        "images_downloaded": success,
        "images_failed": fail,
        "stores": stores,
        "sizes": sizes,
        "failed": fail_list,
    }
    (OUTPUT_DIR / "download_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
