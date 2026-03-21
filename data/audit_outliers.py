"""
Annotation Outlier Detector
----------------------------
For each category, computes a perceptual hash for every bbox crop, then finds
crops that look the most different from the rest of their category.

These are the most likely mislabeled annotations — wrong category_id assigned
to a box that actually contains a different product.

Algorithm:
  1. Resize crop to 16x16 grayscale → 256-bit mean-hash
  2. Compute "centroid hash" = majority-vote across all hashes in the category
  3. Sort crops by Hamming distance from centroid (descending)
  4. Flag top K outliers per category + any crop exceeding a hard distance threshold

Output:
  data/audit_out/outliers.html  — gallery sorted by worst category first
  data/audit_out/hash_cache.json — cached crop hashes (fast reruns)

Usage:
  python3 data/audit_outliers.py              # all categories with ≥5 annotations
  python3 data/audit_outliers.py --min-anns 10 --top-cats 50
  python3 data/audit_outliers.py --cats 80,105,267
"""

import argparse
import base64
import json
import random
from collections import defaultdict
from io import BytesIO
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Install Pillow: pip install Pillow")

# ─── Paths ────────────────────────────────────────────────────────────────────

ANNOTATIONS_PATH = Path.home() / "Downloads" / "train" / "annotations.json"
IMAGES_DIR       = Path.home() / "Downloads" / "train" / "images"
PRODUCT_IMAGES   = Path.home() / "Downloads" / "NM_NGD_product_images"
CATEGORY_MAPPING = Path(__file__).parent / "category_mapping.json"
KASSAL_IMAGES    = Path(__file__).parent / "audit_out" / "kassal_images"
HASH_CACHE_PATH  = Path(__file__).parent / "audit_out" / "hash_cache.json"
OUT_DIR          = Path(__file__).parent / "audit_out"

# ─── Config ───────────────────────────────────────────────────────────────────

HASH_SIZE        = 16      # 16x16 = 256 bits
OUTLIER_K        = 5       # max outliers shown per category
HARD_THRESHOLD   = 0.30    # flag if Hamming > 30% of bits differ from centroid
CROP_PADDING     = 6
THUMB_DIM        = 180
REF_DIM          = 180
MIN_CROP_PX      = 20      # skip crops smaller than this in either dimension

# ─── Hash functions ───────────────────────────────────────────────────────────

def mean_hash(img):
    """256-bit mean hash as Python int. Returns None if image too small."""
    w, h = img.size
    if w < MIN_CROP_PX or h < MIN_CROP_PX:
        return None
    small = img.convert("L").resize((HASH_SIZE, HASH_SIZE), Image.LANCZOS)
    pixels = list(small.getdata())
    avg = sum(pixels) / len(pixels)
    bits = [1 if p >= avg else 0 for p in pixels]
    val = 0
    for b in bits:
        val = (val << 1) | b
    return val


def hamming(h1, h2):
    """Hamming distance as fraction of total bits."""
    return bin(h1 ^ h2).count("1") / (HASH_SIZE * HASH_SIZE)


def centroid_hash(hashes):
    """Majority-vote hash: bit i = 1 if more than half of hashes have it set."""
    n = len(hashes)
    total_bits = HASH_SIZE * HASH_SIZE
    counts = [0] * total_bits
    for h in hashes:
        for i in range(total_bits):
            if (h >> (total_bits - 1 - i)) & 1:
                counts[i] += 1
    val = 0
    for c in counts:
        val = (val << 1) | (1 if c > n / 2 else 0)
    return val


# ─── Image helpers ────────────────────────────────────────────────────────────

def load_crop(shelf_path, bbox):
    try:
        img = Image.open(shelf_path)
    except Exception:
        return None
    x, y, w, h = [int(v) for v in bbox]
    iw, ih = img.size
    x1 = max(0, x - CROP_PADDING)
    y1 = max(0, y - CROP_PADDING)
    x2 = min(iw, x + w + CROP_PADDING)
    y2 = min(ih, y + h + CROP_PADDING)
    if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
        return None
    return img.crop((x1, y1, x2, y2))


def thumb_b64(img, max_dim=THUMB_DIM):
    if img is None:
        return None
    w, h = img.size
    scale = max_dim / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode()


def best_ref(product_code):
    if not product_code:
        return None
    folder = PRODUCT_IMAGES / product_code
    if not folder.exists():
        return None
    for n in ("front.jpg", "main.jpg", "back.jpg"):
        p = folder / n
        if p.exists():
            return p
    jpgs = list(folder.glob("*.jpg"))
    return jpgs[0] if jpgs else None


def kassal_img(product_code):
    if not product_code:
        return None
    p = KASSAL_IMAGES / f"{product_code}.jpg"
    return p if p.exists() else None


# ─── Core ─────────────────────────────────────────────────────────────────────

def compute_all_hashes(annotations, img_map, hash_cache):
    """Compute/load hashes for every annotation. Returns {ann_id: hash_int|None}."""
    result = {}
    needs_compute = [a for a in annotations if str(a["id"]) not in hash_cache]
    total = len(needs_compute)

    if total:
        print(f"Computing hashes for {total} crops (cached: {len(annotations)-total}) …")

    for i, ann in enumerate(needs_compute):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{total} …")
        shelf_file = img_map.get(ann["image_id"])
        if not shelf_file:
            hash_cache[str(ann["id"])] = None
            continue
        crop = load_crop(IMAGES_DIR / shelf_file, ann["bbox"])
        h = mean_hash(crop) if crop is not None else None
        hash_cache[str(ann["id"])] = h

    for ann in annotations:
        result[ann["id"]] = hash_cache.get(str(ann["id"]))

    return result


def find_outliers(ann_hashes, ann_list):
    """
    Given hashes for one category, return list of (ann, distance) sorted
    by distance descending. Distance is Hamming fraction vs centroid.
    """
    valid = [(a, ann_hashes[a["id"]]) for a in ann_list if ann_hashes.get(a["id"]) is not None]
    if len(valid) < 3:
        return []

    hashes = [h for _, h in valid]
    centroid = centroid_hash(hashes)

    scored = [(a, hamming(h, centroid)) for a, h in valid]
    scored.sort(key=lambda x: -x[1])
    return scored


# ─── HTML builder ─────────────────────────────────────────────────────────────

def build_html(cat_sections):
    body = "\n".join(cat_sections)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Annotation Outlier Detector</title>
<style>
  body   {{ font-family: Arial, sans-serif; background: #1a1a2e; color: #eee;
            margin: 0; padding: 16px; }}
  h1     {{ color: #e94560; }}
  section.cat {{ background: #16213e; border-radius: 8px; margin: 14px 0;
                 padding: 14px; border-left: 4px solid #0f3460; }}
  .cat-hdr h2 {{ margin: 0 0 4px; font-size: 1.05em; color: #e94560; }}
  .cat-meta   {{ font-size: 0.78em; color: #aaa; margin-bottom: 10px; }}
  .row    {{ display: flex; gap: 14px; align-items: flex-start; flex-wrap: wrap; }}
  .ref-panel  {{ min-width: 100px; text-align: center; }}
  .ref-panel img {{ max-width: {REF_DIM}px; border-radius: 4px;
                    border: 2px solid #0f3460; }}
  .panel-lbl  {{ font-size: 0.72em; color: #888; margin-bottom: 4px; }}
  .crops-panel {{ flex: 1; }}
  .crops-row  {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .crop-card  {{ background: #0f3460; border-radius: 6px; padding: 6px;
                 text-align: center; position: relative; }}
  .crop-card img {{ max-width: {THUMB_DIM}px; max-height: {THUMB_DIM}px;
                    display: block; border-radius: 3px; }}
  .crop-meta  {{ font-size: 0.65em; color: #ccc; margin-top: 3px; line-height: 1.4; }}
  .dist-bar   {{ height: 4px; border-radius: 2px; margin-top: 3px; }}
  .no-img     {{ width: 100px; height: 70px; background: #0f3460; border-radius: 4px;
                 display: flex; align-items: center; justify-content: center;
                 font-size: 0.72em; color: #666; }}
  .badge      {{ border-radius: 4px; padding: 2px 6px; font-size: 0.72em;
                 font-weight: bold; margin-left: 6px; }}
  .red    {{ background: #c0392b; color: #fff; }}
  .orange {{ background: #e67e22; color: #fff; }}
  .yellow {{ background: #f39c12; color: #000; }}
  .green  {{ background: #27ae60; color: #fff; }}
</style>
</head>
<body>
<h1>Annotation Outlier Detector</h1>
<p style="color:#aaa">
  Crops sorted by visual distance from their category centroid (highest = most suspicious).
  These are the most likely mislabeled annotations.<br>
  Hover over a crop for annotation_id, image_id, and Hamming distance.
  Categories sorted by worst outlier distance first.
</p>
{body}
</body>
</html>"""


def dist_color(d):
    if d >= 0.40:
        return "#c0392b"
    if d >= 0.30:
        return "#e67e22"
    if d >= 0.20:
        return "#f39c12"
    return "#27ae60"


def cat_badge(max_dist):
    if max_dist >= 0.40:
        return '<span class="badge red">HIGH</span>'
    if max_dist >= 0.30:
        return '<span class="badge orange">MEDIUM</span>'
    if max_dist >= 0.20:
        return '<span class="badge yellow">LOW</span>'
    return '<span class="badge green">OK</span>'


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cats",      default=None, help="Comma-separated category IDs")
    parser.add_argument("--min-anns",  type=int, default=5, help="Min annotations per category (default 5)")
    parser.add_argument("--top-cats",  type=int, default=0, help="Show only top N worst categories")
    parser.add_argument("--top-k",     type=int, default=OUTLIER_K, help="Outliers shown per category")
    parser.add_argument("--threshold", type=float, default=HARD_THRESHOLD, help="Hard outlier threshold (0-1)")
    parser.add_argument("--output",    default=None)
    args = parser.parse_args()

    print("Loading annotations …")
    data     = json.load(open(ANNOTATIONS_PATH))
    cats     = {c["id"]: c["name"] for c in data["categories"]}
    img_map  = {i["id"]: i["file_name"] for i in data["images"]}
    mapping  = {}
    if CATEGORY_MAPPING.exists():
        mapping = {int(k): v for k, v in json.load(open(CATEGORY_MAPPING)).items()}

    ann_by_cat = defaultdict(list)
    for a in data["annotations"]:
        ann_by_cat[a["category_id"]].append(a)

    # Load / init hash cache
    hash_cache = {}
    if HASH_CACHE_PATH.exists():
        hash_cache = json.load(open(HASH_CACHE_PATH))
        # Convert stored ints back from JSON strings
        for k, v in hash_cache.items():
            if isinstance(v, str):
                hash_cache[k] = int(v) if v else None

    # Determine which categories to process
    if args.cats:
        process_ids = [int(c) for c in args.cats.split(",")]
    else:
        process_ids = [cid for cid, anns in ann_by_cat.items() if len(anns) >= args.min_anns]

    # Gather all annotations for hashing
    all_anns = [a for a in data["annotations"] if a["category_id"] in set(process_ids)]
    ann_hashes = compute_all_hashes(all_anns, img_map, hash_cache)

    # Save updated cache
    HASH_CACHE_PATH.parent.mkdir(exist_ok=True)
    json.dump(hash_cache, open(HASH_CACHE_PATH, "w"))

    # Find outliers per category
    cat_results = []
    for cid in process_ids:
        anns = ann_by_cat[cid]
        if len(anns) < args.min_anns:
            continue
        scored = find_outliers(ann_hashes, anns)
        if not scored:
            continue
        max_dist = scored[0][1]
        outliers = [s for s in scored if s[1] >= args.threshold][:args.top_k]
        if not outliers:
            outliers = scored[:min(2, len(scored))]  # always show at least 2

        cat_results.append({
            "cat_id":   cid,
            "name":     cats.get(cid, f"cat_{cid}"),
            "info":     mapping.get(cid, {}),
            "total":    len(anns),
            "max_dist": max_dist,
            "outliers": outliers,
        })

    # Sort by worst outlier distance
    cat_results.sort(key=lambda x: -x["max_dist"])

    if args.top_cats:
        cat_results = cat_results[:args.top_cats]

    print(f"Building HTML for {len(cat_results)} categories …")

    sections = []
    for cr in cat_results:
        cid    = cr["cat_id"]
        name   = cr["name"]
        info   = cr["info"]
        code   = info.get("product_code")
        total  = cr["total"]

        ref_img    = None
        ref_path   = best_ref(code)
        if ref_path:
            try:
                ref_img = Image.open(ref_path)
            except Exception:
                pass
        kas_path = kassal_img(code)
        kas_img  = None
        if kas_path:
            try:
                kas_img = Image.open(kas_path)
            except Exception:
                pass

        ref_html = (f'<img src="data:image/jpeg;base64,{thumb_b64(ref_img)}" title="Reference catalog image">'
                    if ref_img else '<div class="no-img">no ref image</div>')
        kas_html = (f'<img src="data:image/jpeg;base64,{thumb_b64(kas_img)}" title="Kassal store image">'
                    if kas_img else '<div class="no-img">no kassal image</div>')

        crop_cards = []
        for ann, dist in cr["outliers"]:
            shelf_file = img_map.get(ann["image_id"])
            if not shelf_file:
                continue
            crop = load_crop(IMAGES_DIR / shelf_file, ann["bbox"])
            if crop is None:
                continue
            b64 = thumb_b64(crop)
            if b64 is None:
                continue
            bx, by, bw, bh = ann["bbox"]
            ann_id = ann["id"]
            img_id = ann["image_id"]
            dist_pct = int(dist * 100)
            color = dist_color(dist)
            title = f"ann_id={ann_id} img_id={img_id} dist={dist:.3f} bbox=[{bx},{by},{bw},{bh}]"
            crop_cards.append(
                f'<div class="crop-card">'
                f'<img src="data:image/jpeg;base64,{b64}" title="{title}">'
                f'<div class="crop-meta">ann#{ann_id}<br>img#{img_id}<br>dist={dist:.3f}</div>'
                f'<div class="dist-bar" style="width:{min(100,dist_pct*2)}%;background:{color}"></div>'
                f'</div>'
            )

        if not crop_cards:
            continue

        badge = cat_badge(cr["max_dist"])
        sections.append(f"""
<section class="cat">
  <div class="cat-hdr">
    <h2>cat #{cid} — {name} {badge}</h2>
    <div class="cat-meta">
      code: {code or "—"} &nbsp;|&nbsp;
      total annotations: {total} &nbsp;|&nbsp;
      max outlier dist: {cr["max_dist"]:.3f} &nbsp;|&nbsp;
      showing top {len(crop_cards)} outliers
    </div>
  </div>
  <div class="row">
    <div class="ref-panel">
      <div class="panel-lbl">Reference (catalog)</div>
      {ref_html}
    </div>
    <div class="ref-panel">
      <div class="panel-lbl">Kassal store image</div>
      {kas_html}
    </div>
    <div class="crops-panel">
      <div class="panel-lbl">Most suspicious crops (highest Hamming distance from category centroid)</div>
      <div class="crops-row">{"".join(crop_cards)}</div>
    </div>
  </div>
</section>""")

    html = build_html(sections)
    out = Path(args.output) if args.output else OUT_DIR / "outliers.html"
    out.write_text(html, encoding="utf-8")
    print(f"\nSaved → {out}  ({len(sections)} categories)")

    # Print top 20 worst for quick CLI review
    print("\nTop 20 categories by max outlier distance:")
    print(f"{'cat_id':>7} {'dist':>6}  name")
    for cr in cat_results[:20]:
        print(f"  #{cr['cat_id']:>4}  {cr['max_dist']:.3f}  {cr['name']}")

    print(f"\nopen '{out}'")


if __name__ == "__main__":
    main()
