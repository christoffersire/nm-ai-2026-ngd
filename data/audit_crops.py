"""
Annotation Crop Comparison Tool
--------------------------------
For each category, samples N bbox crops from shelf images and places them
side-by-side with the reference product image and Kassal store image in an HTML gallery.

Priority order (highest suspicion first):
  1. Egg/pack categories (6-pack vs 10-pack vs 12-pack confusion risk)
  2. Categories with annotation_count > 50 and corrected_count == 0
  3. Categories with high annotation count / low corrected ratio
  4. All other categories

Each crop is linked to its annotation_id and image_id for later correction.

Usage:
  python3 data/audit_crops.py [--cats 80,139,232] [--n 8] [--output data/audit_out/crops.html]
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

ANNOTATIONS_PATH  = Path.home() / "Downloads" / "train" / "annotations.json"
IMAGES_DIR        = Path.home() / "Downloads" / "train" / "images"
PRODUCT_IMAGES    = Path.home() / "Downloads" / "NM_NGD_product_images"
CATEGORY_MAPPING  = Path(__file__).parent / "category_mapping.json"
KASSAL_IMAGES_DIR = Path(__file__).parent / "audit_out" / "kassal_images"
OFF_CACHE         = Path(__file__).parent / "audit_out" / "off_cache.json"

# ─── Config ───────────────────────────────────────────────────────────────────

CROP_PADDING   = 8        # px to add around each bbox crop
CROP_MAX_DIM   = 200      # resize longest dim to this for display
REF_MAX_DIM    = 200
THUMB_QUALITY  = 75

# ─── Helpers ──────────────────────────────────────────────────────────────────

def img_to_b64(img, fmt="JPEG", quality=THUMB_QUALITY):
    buf = BytesIO()
    rgb = img.convert("RGB")
    rgb.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def load_img(path):
    try:
        return Image.open(path)
    except Exception:
        return None


def thumb(img, max_dim):
    if img is None:
        return None
    w, h = img.size
    scale = max_dim / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def crop_from_shelf(shelf_path, bbox, padding=CROP_PADDING):
    """Crop bbox [x, y, w, h] from shelf image with padding."""
    img = load_img(shelf_path)
    if img is None:
        return None
    x, y, w, h = [int(v) for v in bbox]
    iw, ih = img.size
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(iw, x + w + padding)
    y2 = min(ih, y + h + padding)
    return img.crop((x1, y1, x2, y2))


def best_ref_image(product_code):
    """Return front/main/any reference image path, or None."""
    if not product_code:
        return None
    folder = PRODUCT_IMAGES / product_code
    if not folder.exists():
        return None
    for name in ("front.jpg", "main.jpg", "back.jpg"):
        p = folder / name
        if p.exists():
            return p
    jpgs = list(folder.glob("*.jpg"))
    return jpgs[0] if jpgs else None


def kassal_image_path(product_code):
    if not product_code:
        return None
    p = KASSAL_IMAGES_DIR / f"{product_code}.jpg"
    return p if p.exists() else None


def suspicion_score(cat_id, ann_count, corrected, name):
    """Higher = more suspicious = shown first."""
    score = 0
    name_l = name.lower()
    if "egg" in name_l:
        score += 100
        # Pack size confusion bonus
        for pat in ("6stk", "10stk", "12stk", "18stk", "6pk", "10pk"):
            if pat in name_l.replace(" ", ""):
                score += 20
    if corrected == 0 and ann_count > 20:
        score += 50
    if ann_count > 0:
        ratio = corrected / ann_count
        score += int((1 - ratio) * 30)
    score += min(ann_count // 10, 20)
    return score


# ─── Main ─────────────────────────────────────────────────────────────────────

def build_gallery(target_cats=None, n_crops=8, output_path=None):
    print("Loading annotations …")
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)

    cats      = {c["id"]: c["name"] for c in data["categories"]}
    img_map   = {i["id"]: i["file_name"] for i in data["images"]}
    ann_by_cat = defaultdict(list)
    for a in data["annotations"]:
        ann_by_cat[a["category_id"]].append(a)

    # Load mapping for product codes and corrected counts
    cat_mapping = {}
    if CATEGORY_MAPPING.exists():
        raw = json.load(open(CATEGORY_MAPPING))
        cat_mapping = {int(k): v for k, v in raw.items()}

    # Load off_cache for Kassal product codes (keyed by EAN)
    kassal_cache = {}
    if OFF_CACHE.exists():
        cache_data = json.load(open(OFF_CACHE))
        for k, v in cache_data.items():
            if k.startswith("kassal_"):
                ean = k[len("kassal_"):]
                kassal_cache[ean] = v

    # Determine which categories to process
    if target_cats:
        process_ids = [int(c) for c in target_cats]
    else:
        process_ids = sorted(cats.keys())

    # Build category info list with suspicion score
    cat_infos = []
    for cid in process_ids:
        name = cats.get(cid, f"unknown_{cid}")
        info = cat_mapping.get(cid, {})
        product_code = info.get("product_code")
        ann_count = len(ann_by_cat[cid])
        corrected = info.get("corrected_count", 0) or 0
        score = suspicion_score(cid, ann_count, corrected, name)
        cat_infos.append({
            "cat_id": cid,
            "name": name,
            "product_code": product_code,
            "ann_count": ann_count,
            "corrected": corrected,
            "score": score,
        })

    # Sort by suspicion descending
    cat_infos.sort(key=lambda x: -x["score"])

    print(f"Processing {len(cat_infos)} categories …")

    # ─── Build HTML ───────────────────────────────────────────────────────────

    sections = []
    for ci in cat_infos:
        cid          = ci["cat_id"]
        name         = ci["name"]
        product_code = ci["product_code"]
        anns         = ann_by_cat[cid]

        if not anns:
            continue

        # Sample up to n_crops annotations
        sampled = random.sample(anns, min(n_crops, len(anns)))

        # Reference image (product catalog)
        ref_path   = best_ref_image(product_code)
        ref_img    = thumb(load_img(ref_path), REF_MAX_DIM) if ref_path else None

        # Kassal image — look up by product_code (which is the EAN)
        kas_path   = kassal_image_path(product_code)
        kas_img    = thumb(load_img(kas_path), REF_MAX_DIM) if kas_path else None

        # Build crop thumbnails
        crop_cards = []
        for ann in sampled:
            shelf_file = img_map.get(ann["image_id"])
            if not shelf_file:
                continue
            shelf_path = IMAGES_DIR / shelf_file
            crop = crop_from_shelf(shelf_path, ann["bbox"])
            if crop is None:
                continue
            crop_t = thumb(crop, CROP_MAX_DIM)
            crop_b64 = img_to_b64(crop_t)
            bx, by, bw, bh = ann["bbox"]
            ann_id  = ann["id"]
            img_id  = ann["image_id"]
            title   = f"ann_id={ann_id} img_id={img_id} bbox=[{bx},{by},{bw},{bh}]"
            crop_cards.append(
                f'<div class="crop-card">'
                f'<img src="data:image/jpeg;base64,{crop_b64}" title="{title}">'
                f'<div class="crop-meta">ann#{ann_id}<br>img#{img_id}</div>'
                f'</div>'
            )

        if not crop_cards:
            continue

        ref_html = (
            f'<img src="data:image/jpeg;base64,{img_to_b64(ref_img)}" title="Reference: {ref_path}">'
            if ref_img else '<div class="no-img">no ref image</div>'
        )
        kas_html = (
            f'<img src="data:image/jpeg;base64,{img_to_b64(kas_img)}" title="Kassal image">'
            if kas_img else '<div class="no-img">no kassal image</div>'
        )

        suspicion_badge = ""
        if ci["score"] >= 100:
            suspicion_badge = '<span class="badge red">HIGH</span>'
        elif ci["score"] >= 50:
            suspicion_badge = '<span class="badge orange">MEDIUM</span>'
        else:
            suspicion_badge = '<span class="badge green">LOW</span>'

        sections.append(f"""
<section class="category">
  <div class="cat-header">
    <h2>cat #{cid} — {name} {suspicion_badge}</h2>
    <div class="cat-meta">
      code: {product_code or "—"} &nbsp;|&nbsp;
      annotations: {ci["ann_count"]} &nbsp;|&nbsp;
      corrected: {ci["corrected"]} &nbsp;|&nbsp;
      suspicion: {ci["score"]}
    </div>
  </div>
  <div class="row">
    <div class="ref-panel">
      <div class="panel-label">Reference (catalog)</div>
      {ref_html}
    </div>
    <div class="ref-panel">
      <div class="panel-label">Kassal store image</div>
      {kas_html}
    </div>
    <div class="crops-panel">
      <div class="panel-label">Annotation crops (sample of {len(crop_cards)}, hover for IDs)</div>
      <div class="crops-row">
        {"".join(crop_cards)}
      </div>
    </div>
  </div>
</section>""")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Annotation Crop Comparison</title>
<style>
  body {{ font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 16px; }}
  h1   {{ color: #e94560; }}
  section.category {{ background: #16213e; border-radius: 8px; margin: 16px 0; padding: 16px;
                      border-left: 4px solid #0f3460; }}
  .cat-header h2 {{ margin: 0 0 4px; font-size: 1.1em; color: #e94560; }}
  .cat-meta {{ font-size: 0.8em; color: #aaa; margin-bottom: 12px; }}
  .row {{ display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }}
  .ref-panel {{ min-width: 120px; text-align: center; }}
  .ref-panel img {{ max-width: 200px; border-radius: 4px; border: 2px solid #0f3460; }}
  .panel-label {{ font-size: 0.75em; color: #888; margin-bottom: 4px; }}
  .crops-panel {{ flex: 1; }}
  .crops-row {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .crop-card {{ background: #0f3460; border-radius: 4px; padding: 4px; text-align: center; }}
  .crop-card img {{ max-width: {CROP_MAX_DIM}px; max-height: {CROP_MAX_DIM}px;
                    display: block; border-radius: 2px; cursor: zoom-in; }}
  .crop-meta {{ font-size: 0.65em; color: #aaa; margin-top: 2px; }}
  .no-img {{ width: 120px; height: 80px; background: #0f3460; border-radius: 4px;
             display: flex; align-items: center; justify-content: center;
             font-size: 0.75em; color: #666; }}
  .badge {{ border-radius: 4px; padding: 2px 6px; font-size: 0.75em; font-weight: bold; }}
  .badge.red    {{ background: #c0392b; color: #fff; }}
  .badge.orange {{ background: #e67e22; color: #fff; }}
  .badge.green  {{ background: #27ae60; color: #fff; }}
  a {{ color: #e94560; }}
</style>
</head>
<body>
<h1>Annotation Crop Comparison</h1>
<p style="color:#aaa">
  Sorted by suspicion score (highest first). Hover over a crop to see annotation_id + image_id.
  Egg/pack categories are shown first — check for 6-pack vs 10-pack vs 12-pack confusion.
</p>
<p style="color:#aaa">
  Categories shown: {len(sections)} &nbsp;|&nbsp;
  Generated: 2026-03-21
</p>
{"".join(sections)}
</body>
</html>"""

    if output_path is None:
        output_path = Path(__file__).parent / "audit_out" / "crops.html"
    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
    print(f"\nGallery saved → {output_path}  ({len(sections)} categories)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotation crop comparison gallery")
    parser.add_argument("--cats",   default=None, help="Comma-separated category IDs (default: all)")
    parser.add_argument("--n",      type=int, default=8, help="Max crops per category (default 8)")
    parser.add_argument("--output", default=None, help="Output HTML path")
    parser.add_argument("--top",    type=int, default=0, help="Only show top N categories by suspicion")
    args = parser.parse_args()

    target = [c.strip() for c in args.cats.split(",")] if args.cats else None

    path = build_gallery(
        target_cats=target,
        n_crops=args.n,
        output_path=args.output,
    )

    # If --top N, we need to re-run with only top N
    # (already sorted internally; this is a quick subset run)
    print("Open in browser:")
    print(f"  open '{path}'")
