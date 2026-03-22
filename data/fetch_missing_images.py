"""
Laster ned produktbilder for kategorier som mangler bildemapper
i NM_NGD_product_images/.

Kjøres én gang etter at dere har hentet ned konkurransedataene.

Usage:
  python3 data/fetch_missing_images.py
"""

import json
import time
import urllib.request
from pathlib import Path

PRODUCT_IMAGES  = Path.home() / "Downloads" / "NM_NGD_product_images"
CATEGORY_MAPPING = Path(__file__).parent / "category_mapping.json"
CONFIG_PATH     = Path(__file__).parent / "audit_config.json"

USER_AGENT = "NM-AI-2026-DataAudit/1.0 (educational)"

# Produkter som mangler og vet vi kan hente
MISSING = [
    {
        "ean":  "7310130011566",
        "name": "AXA Gold Müsli Frukt 750g",
        "images": {
            "front": "https://res.cloudinary.com/norgesgruppen/image/upload/v1646957245/Product/7310130011566.jpg",
            "main":  "https://images.openfoodfacts.org/images/products/731/013/001/1566/front_fr.6.400.jpg",
            "back":  "https://images.openfoodfacts.org/images/products/731/013/001/1566/ingredients_fr.21.400.jpg",
        }
    },
]


def fetch(url, dest: Path):
    if dest.exists():
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as r:
            dest.write_bytes(r.read())
        time.sleep(0.5)
        return True
    except Exception as e:
        print(f"    FEIL: {e}")
        return False


def main():
    PRODUCT_IMAGES.mkdir(parents=True, exist_ok=True)

    for product in MISSING:
        ean    = product["ean"]
        name   = product["name"]
        folder = PRODUCT_IMAGES / ean
        folder.mkdir(exist_ok=True)

        print(f"{ean}  {name}")
        for img_type, url in product["images"].items():
            dest = folder / f"{img_type}.jpg"
            ok = fetch(url, dest)
            status = f"{dest.stat().st_size//1024}KB" if ok else "FEIL"
            print(f"  {img_type}.jpg: {status}")

    print("\nFerdig.")


if __name__ == "__main__":
    main()
