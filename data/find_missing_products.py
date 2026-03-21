"""
Find Missing Product Codes + Discontinued Category Matcher
------------------------------------------------------------
Two problems solved, both output SUGGESTIONS ONLY — nothing is changed automatically.

Problem 1 — 33 categories have no product code in category_mapping.json:
  → Search Kassal by product name to find candidate EANs
  → Score by name similarity + cross-check against Open Food Facts

Problem 2 — Discontinued categories (AXA Müsli) can't stay as-is:
  → Use perceptual hash centroids (from hash_cache.json) to find the most
    visually similar surviving category
  → Also suggest by name similarity as a second signal

Output:
  data/audit_out/missing_products.csv   — one row per category, columns: suggestion_ean,
                                          suggestion_name, name_sim, source, notes
  data/audit_out/missing_products.html  — visual gallery with Kassal images + crops
  (nothing written to annotations.json or category_mapping.json)

Usage:
  python3 data/find_missing_products.py
  python3 data/find_missing_products.py --skip-kassal   # offline, hash-only
"""

import argparse
import base64
import csv
import json
import random
import time
import urllib.error
import urllib.request
from collections import defaultdict
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("pip install Pillow")

# ─── Paths ────────────────────────────────────────────────────────────────────

ANNOTATIONS_PATH = Path.home() / "Downloads" / "train" / "annotations.json"
IMAGES_DIR       = Path.home() / "Downloads" / "train" / "images"
PRODUCT_IMAGES   = Path.home() / "Downloads" / "NM_NGD_product_images"
CATEGORY_MAPPING = Path(__file__).parent / "category_mapping.json"
HASH_CACHE       = Path(__file__).parent / "audit_out" / "hash_cache.json"
OFF_CACHE        = Path(__file__).parent / "audit_out" / "off_cache.json"
CONFIG_PATH      = Path(__file__).parent / "audit_config.json"
OUT_DIR          = Path(__file__).parent / "audit_out"

KASSAL_SEARCH  = "https://kassal.app/api/v1/products?search={q}&size=10"
KASSAL_RATE_S  = 1.1   # 60 req/min limit
USER_AGENT     = "nm-ai-2026-ngd-audit/1.0"
HASH_SIZE      = 16    # must match audit_outliers.py
CROP_PADDING   = 6
THUMB_DIM      = 160
N_CROP_SAMPLES = 999   # show all crops (overridable via --n-crops)

# ─── Load config ──────────────────────────────────────────────────────────────

def load_token():
    if CONFIG_PATH.exists():
        cfg = json.load(open(CONFIG_PATH))
        return cfg.get("kassal_token") or ""
    return ""

# ─── Kassal search ────────────────────────────────────────────────────────────

def kassal_search(query, token, cache):
    cache_key = f"search_{query.lower().strip()}"
    if cache_key in cache:
        return cache[cache_key]
    url = KASSAL_SEARCH.format(q=urllib.parse.quote(query))
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/json",
                 "User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())
        products = data.get("data") or []
        result = [
            {
                "ean":   p.get("ean") or "",
                "name":  p.get("name") or "",
                "store": p.get("store", {}).get("name", "") if p.get("store") else "",
                "image": p.get("image") or "",
                "url":   p.get("url") or "",
            }
            for p in products if p.get("ean")
        ]
    except urllib.error.HTTPError as e:
        result = []
    cache[cache_key] = result
    return result

import urllib.parse

# ─── Hash helpers ─────────────────────────────────────────────────────────────

def mean_hash(img):
    if img is None:
        return None
    w, h = img.size
    if w < 16 or h < 16:
        return None
    small = img.convert("L").resize((HASH_SIZE, HASH_SIZE), Image.LANCZOS)
    pixels = list(small.getdata())
    avg = sum(pixels) / len(pixels)
    bits = [1 if p >= avg else 0 for p in pixels]
    val = 0
    for b in bits:
        val = (val << 1) | b
    return val


def hamming_frac(h1, h2):
    return bin(h1 ^ h2).count("1") / (HASH_SIZE * HASH_SIZE)


def centroid_hash(hashes):
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


def load_hash_cache():
    if not HASH_CACHE.exists():
        return {}
    raw = json.load(open(HASH_CACHE))
    # JSON stores large ints as-is; None comes back as None
    return {k: (int(v) if isinstance(v, (int, float)) else None) for k, v in raw.items()}


def category_centroid(cid, ann_list, hash_cache):
    hashes = [hash_cache.get(str(a["id"])) for a in ann_list]
    hashes = [h for h in hashes if h is not None]
    if len(hashes) < 2:
        return None
    return centroid_hash(hashes)


# ─── Image helpers ────────────────────────────────────────────────────────────

def load_crop(shelf_path, bbox):
    try:
        img = Image.open(shelf_path)
    except Exception:
        return None
    x, y, w, h = [int(v) for v in bbox]
    iw, ih = img.size
    x1, y1 = max(0, x - CROP_PADDING), max(0, y - CROP_PADDING)
    x2, y2 = min(iw, x + w + CROP_PADDING), min(ih, y + h + CROP_PADDING)
    if x2 - x1 < 16 or y2 - y1 < 16:
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


def url_to_b64(url, token):
    """Fetch a remote image and return b64 thumbnail, or None."""
    if not url:
        return None
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            data = r.read()
        img = Image.open(BytesIO(data))
        return thumb_b64(img)
    except Exception:
        return None


# ─── Name similarity ──────────────────────────────────────────────────────────

def name_sim(a, b):
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def best_match(our_name, candidates):
    """Return (best_candidate_dict, similarity) from a list of {name, ean, ...}"""
    if not candidates:
        return None, 0.0
    ranked = sorted(candidates, key=lambda c: name_sim(our_name, c["name"]), reverse=True)
    best = ranked[0]
    return best, name_sim(our_name, best["name"])


# ─── Suggestion builder ───────────────────────────────────────────────────────

def find_suggestions_for_no_code(cat_id, cat_name, ann_list, img_map,
                                  hash_cache, off_cache, token, skip_kassal):
    """Search Kassal by name, return list of suggestion dicts."""
    suggestions = []

    if not skip_kassal and token:
        results = kassal_search(cat_name, token, off_cache)
        time.sleep(KASSAL_RATE_S)
        for r in results[:5]:
            sim = name_sim(cat_name, r["name"])
            suggestions.append({
                "source":       "kassal_search",
                "ean":          r["ean"],
                "name":         r["name"],
                "name_sim":     round(sim, 3),
                "store":        r["store"],
                "image_url":    r["image"],
                "kassal_url":   r["url"],
                "visual_sim":   None,
                "confidence":   "high" if sim > 0.85 else ("medium" if sim > 0.65 else "low"),
            })

    suggestions.sort(key=lambda s: -s["name_sim"])
    return suggestions


def find_suggestions_for_discontinued(cat_id, cat_name, ann_list, img_map,
                                       hash_cache, all_cat_centroids, all_cats,
                                       mapping, token, off_cache, skip_kassal):
    """
    For a discontinued category, find most visually similar surviving categories
    AND search Kassal by name.
    """
    suggestions = []

    # 1. Visual similarity against all surviving category centroids
    our_centroid = category_centroid(cat_id, ann_list, hash_cache)
    if our_centroid is not None:
        visual_scores = []
        for other_cid, other_centroid in all_cat_centroids.items():
            if other_cid == cat_id or other_centroid is None:
                continue
            dist = hamming_frac(our_centroid, other_centroid)
            sim  = 1.0 - dist
            visual_scores.append((other_cid, sim))
        visual_scores.sort(key=lambda x: -x[1])

        for other_cid, vis_sim in visual_scores[:5]:
            other_name = all_cats.get(other_cid, "")
            other_info = mapping.get(str(other_cid), {})
            name_similarity = name_sim(cat_name, other_name)
            suggestions.append({
                "source":      "visual_hash",
                "target_cat":  other_cid,
                "target_name": other_name,
                "target_code": other_info.get("product_code", ""),
                "visual_sim":  round(vis_sim, 3),
                "name_sim":    round(name_similarity, 3),
                "confidence":  "high" if vis_sim > 0.80 else ("medium" if vis_sim > 0.70 else "low"),
                "ean":         other_info.get("product_code", ""),
                "image_url":   "",
                "kassal_url":  "",
            })

    # 2. Kassal name search as second signal
    if not skip_kassal and token:
        results = kassal_search(cat_name, token, off_cache)
        time.sleep(KASSAL_RATE_S)
        for r in results[:3]:
            sim = name_sim(cat_name, r["name"])
            suggestions.append({
                "source":      "kassal_search",
                "target_cat":  None,
                "target_name": r["name"],
                "target_code": r["ean"],
                "visual_sim":  None,
                "name_sim":    round(sim, 3),
                "confidence":  "high" if sim > 0.80 else ("medium" if sim > 0.60 else "low"),
                "ean":         r["ean"],
                "image_url":   r["image"],
                "kassal_url":  r["url"],
            })

    return suggestions


# ─── HTML builder ─────────────────────────────────────────────────────────────

CONF_COLOR = {"high": "#27ae60", "medium": "#e67e22", "low": "#c0392b"}

def suggestion_chip(s, token):
    conf  = s.get("confidence", "low")
    color = CONF_COLOR.get(conf, "#555")
    src   = s.get("source", "")
    if src == "visual_hash":
        label = f"cat#{s['target_cat']} {s['target_name'][:40]}"
        detail = f"visual={s['visual_sim']:.2f} name={s['name_sim']:.2f}"
    else:
        label  = s.get("target_name") or s.get("name", "")
        label  = label[:45]
        detail = f"EAN={s['ean']} name_sim={s['name_sim']:.2f}"

    img_html = ""
    img_url  = s.get("image_url")
    if img_url:
        b64 = url_to_b64(img_url, token)
        if b64:
            img_html = f'<img src="data:image/jpeg;base64,{b64}" style="max-width:100px;max-height:80px;display:block;margin:4px auto;">'

    return (
        f'<div class="chip" style="border-left:4px solid {color}">'
        f'<div class="chip-src">{src}</div>'
        f'<div class="chip-label">{label}</div>'
        f'<div class="chip-detail">{detail}</div>'
        f'{img_html}'
        f'</div>'
    )


def build_html(sections, total_no_code, total_discontinued):
    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Missing Product Finder</title>
<style>
  body   {{ font-family: Arial, sans-serif; background:#1a1a2e; color:#eee; margin:0; padding:16px; }}
  h1     {{ color:#e94560; }}
  section.cat {{ background:#16213e; border-radius:8px; margin:12px 0; padding:14px;
                 border-left:4px solid #0f3460; }}
  .hdr h2     {{ margin:0 0 4px; font-size:1.05em; color:#e94560; }}
  .meta       {{ font-size:0.78em; color:#aaa; margin-bottom:10px; }}
  .row        {{ display:flex; gap:14px; align-items:flex-start; flex-wrap:wrap; }}
  .crops-panel {{ flex:1; }}
  .crops-row  {{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px; }}
  .crop-card  {{ background:#0f3460; border-radius:4px; padding:4px; text-align:center; }}
  .crop-card img {{ max-width:150px; max-height:150px; display:block; border-radius:3px; }}
  .crop-meta  {{ font-size:0.65em; color:#aaa; margin-top:2px; }}
  .suggestions {{ flex:1; min-width:260px; }}
  .sug-title  {{ font-size:0.8em; color:#aaa; margin-bottom:6px; }}
  .chips      {{ display:flex; flex-wrap:wrap; gap:8px; }}
  .chip       {{ background:#0f3460; border-radius:6px; padding:8px 10px;
                 min-width:200px; max-width:240px; }}
  .chip-src   {{ font-size:0.65em; color:#888; text-transform:uppercase; letter-spacing:.5px; }}
  .chip-label {{ font-size:0.85em; font-weight:bold; margin:2px 0; }}
  .chip-detail {{ font-size:0.72em; color:#aaa; }}
  .type-badge {{ border-radius:4px; padding:2px 8px; font-size:0.72em; font-weight:bold;
                 margin-left:6px; }}
  .disc  {{ background:#c0392b; color:#fff; }}
  .nocode {{ background:#8e44ad; color:#fff; }}
  .ref-panel {{ text-align:center; min-width:120px; }}
  .ref-panel img {{ max-width:140px; border-radius:4px; border:2px solid #0f3460; }}
  .panel-lbl {{ font-size:0.72em; color:#888; margin-bottom:4px; }}
  .no-img {{ width:100px; height:70px; background:#0f3460; border-radius:4px;
             display:flex; align-items:center; justify-content:center;
             font-size:0.72em; color:#666; }}
</style>
</head>
<body>
<h1>Missing Product Finder — Suggestions Only</h1>
<p style="color:#aaa">
  <b style="color:#8e44ad">PURPLE</b> = no product code (33 categories, searched Kassal by name)<br>
  <b style="color:#c0392b">RED</b> = discontinued (2 AXA Müsli, found via visual + name similarity)<br>
  Nothing is applied automatically — verify each suggestion manually before adding to annotation_patches.json.
</p>
<p style="color:#aaa">No-code categories: {total_no_code} &nbsp;|&nbsp; Discontinued: {total_discontinued}</p>
{body}
</body>
</html>"""


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-kassal", action="store_true", help="Skip API calls (offline mode)")
    parser.add_argument("--n-crops", type=int, default=N_CROP_SAMPLES, help="Max crops per category (default: all)")
    args = parser.parse_args()

    print("Loading data …")
    data     = json.load(open(ANNOTATIONS_PATH))
    cats     = {c["id"]: c["name"] for c in data["categories"]}
    img_map  = {i["id"]: i["file_name"] for i in data["images"]}
    mapping  = json.load(open(CATEGORY_MAPPING))
    token    = load_token()
    off_cache = json.load(open(OFF_CACHE)) if OFF_CACHE.exists() else {}
    hash_cache = load_hash_cache()

    ann_by_cat = defaultdict(list)
    for a in data["annotations"]:
        ann_by_cat[a["category_id"]].append(a)

    mapped_ids = set(int(k) for k in mapping.keys())
    all_cat_ids = set(cats.keys())

    # Categories with no metadata entry at all
    no_code_cats = [
        cid for cid in all_cat_ids
        if cid not in mapped_ids and ann_by_cat[cid] and cid != 355
    ]

    # Discontinued (from product_corrections.json)
    corrections_path = OUT_DIR / "product_corrections.json"
    discontinued_ids = []
    if corrections_path.exists():
        corrections = json.load(open(corrections_path))
        discontinued_ids = [c["category_id"] for c in corrections if c["change_type"] == "discontinued"]

    print(f"No-code categories: {len(no_code_cats)}")
    print(f"Discontinued categories: {len(discontinued_ids)}")

    # Pre-compute centroids for all mapped categories (for discontinued matching)
    print("Computing category centroids for visual matching …")
    all_cat_centroids = {}
    for cid, anns in ann_by_cat.items():
        if cid in discontinued_ids or cid == 355:
            continue
        c = category_centroid(cid, anns, hash_cache)
        if c is not None:
            all_cat_centroids[cid] = c

    if not args.skip_kassal and not token:
        print("WARNING: No Kassal token found in audit_config.json — skipping API calls")
        args.skip_kassal = True

    sections = []
    csv_rows = []

    # ── 1. No-code categories ─────────────────────────────────────────────────
    print(f"\nSearching Kassal for {len(no_code_cats)} no-code categories …")
    for i, cid in enumerate(sorted(no_code_cats, key=lambda c: -len(ann_by_cat[c]))):
        name = cats.get(cid, f"cat_{cid}")
        anns = ann_by_cat[cid]
        print(f"  [{i+1}/{len(no_code_cats)}] cat#{cid} ({len(anns)} anns): {name}")

        suggestions = find_suggestions_for_no_code(
            cid, name, anns, img_map, hash_cache, off_cache, token, args.skip_kassal
        )

        # Sample crops
        sampled = random.sample(anns, min(args.n_crops, len(anns)))
        crop_cards = []
        for ann in sampled:
            shelf_file = img_map.get(ann["image_id"])
            if not shelf_file:
                continue
            crop = load_crop(IMAGES_DIR / shelf_file, ann["bbox"])
            b64 = thumb_b64(crop)
            if b64:
                ann_id, img_id = ann["id"], ann["image_id"]
                crop_cards.append(
                    f'<div class="crop-card">'
                    f'<img src="data:image/jpeg;base64,{b64}" title="ann#{ann_id} img#{img_id}">'
                    f'<div class="crop-meta">ann#{ann_id}<br>img#{img_id}</div>'
                    f'</div>'
                )

        chips = "".join(suggestion_chip(s, token) for s in suggestions[:4])
        best  = suggestions[0] if suggestions else {}
        sections.append(f"""
<section class="cat">
  <div class="hdr">
    <h2>cat #{cid} — {name} <span class="type-badge nocode">NO CODE</span></h2>
    <div class="meta">annotations: {len(anns)} &nbsp;|&nbsp; top suggestion: {best.get('target_name') or best.get('name','—')[:50]}</div>
  </div>
  <div class="row">
    <div class="crops-panel">
      <div class="panel-lbl">Sample crops</div>
      <div class="crops-row">{"".join(crop_cards)}</div>
    </div>
    <div class="suggestions">
      <div class="sug-title">Kassal suggestions (verify manually):</div>
      <div class="chips">{chips}</div>
    </div>
  </div>
</section>""")

        for s in suggestions[:3]:
            csv_rows.append({
                "category_id":    cid,
                "category_name":  name,
                "annotation_count": len(anns),
                "problem_type":   "no_product_code",
                "source":         s.get("source"),
                "suggested_ean":  s.get("ean", ""),
                "suggested_name": s.get("target_name") or s.get("name", ""),
                "name_sim":       s.get("name_sim", ""),
                "visual_sim":     s.get("visual_sim", ""),
                "confidence":     s.get("confidence", ""),
                "target_cat_id":  s.get("target_cat", ""),
                "kassal_url":     s.get("kassal_url", ""),
                "notes":          "",
            })

    # ── 2. Discontinued categories ────────────────────────────────────────────
    print(f"\nMatching {len(discontinued_ids)} discontinued categories …")
    for cid in discontinued_ids:
        name = cats.get(cid, f"cat_{cid}")
        anns = ann_by_cat[cid]
        print(f"  cat#{cid} ({len(anns)} anns): {name}")

        suggestions = find_suggestions_for_discontinued(
            cid, name, anns, img_map, hash_cache,
            all_cat_centroids, cats, mapping, token, off_cache, args.skip_kassal
        )

        sampled = random.sample(anns, min(args.n_crops, len(anns)))
        crop_cards = []
        for ann in sampled:
            shelf_file = img_map.get(ann["image_id"])
            if not shelf_file:
                continue
            crop = load_crop(IMAGES_DIR / shelf_file, ann["bbox"])
            b64 = thumb_b64(crop)
            if b64:
                ann_id, img_id = ann["id"], ann["image_id"]
                crop_cards.append(
                    f'<div class="crop-card">'
                    f'<img src="data:image/jpeg;base64,{b64}" title="ann#{ann_id} img#{img_id}">'
                    f'<div class="crop-meta">ann#{ann_id}<br>img#{img_id}</div>'
                    f'</div>'
                )

        chips = "".join(suggestion_chip(s, token) for s in suggestions[:5])
        best  = suggestions[0] if suggestions else {}

        info  = mapping.get(str(cid), {})
        code  = info.get("product_code")
        ref_path = best_ref(code)
        ref_html = (
            f'<div class="ref-panel"><div class="panel-lbl">Our reference</div>'
            f'<img src="data:image/jpeg;base64,{thumb_b64(Image.open(ref_path))}">'
            f'</div>'
        ) if ref_path else ""

        sections.append(f"""
<section class="cat">
  <div class="hdr">
    <h2>cat #{cid} — {name} <span class="type-badge disc">DISCONTINUED</span></h2>
    <div class="meta">annotations: {len(anns)} &nbsp;|&nbsp; EAN: {code or "—"} &nbsp;|&nbsp; best match: {best.get('target_name','—')[:50]}</div>
  </div>
  <div class="row">
    {ref_html}
    <div class="crops-panel">
      <div class="panel-lbl">Shelf crops of discontinued product</div>
      <div class="crops-row">{"".join(crop_cards)}</div>
    </div>
    <div class="suggestions">
      <div class="sug-title">Suggested replacement category (verify visually):</div>
      <div class="chips">{chips}</div>
    </div>
  </div>
</section>""")

        for s in suggestions[:3]:
            csv_rows.append({
                "category_id":    cid,
                "category_name":  name,
                "annotation_count": len(anns),
                "problem_type":   "discontinued",
                "source":         s.get("source"),
                "suggested_ean":  s.get("ean", ""),
                "suggested_name": s.get("target_name") or s.get("name", ""),
                "name_sim":       s.get("name_sim", ""),
                "visual_sim":     s.get("visual_sim", ""),
                "confidence":     s.get("confidence", ""),
                "target_cat_id":  s.get("target_cat", ""),
                "kassal_url":     s.get("kassal_url", ""),
                "notes":          "",
            })

    # ── Write outputs ─────────────────────────────────────────────────────────
    OUT_DIR.mkdir(exist_ok=True)

    # Save updated cache
    json.dump(off_cache, open(OFF_CACHE, "w"), ensure_ascii=False)

    suffix = "_v2" if args.skip_kassal else ""
    csv_path = OUT_DIR / f"missing_products{suffix}.csv"
    html_path = OUT_DIR / f"missing_products{suffix}.html"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        cols = ["category_id","category_name","annotation_count","problem_type",
                "source","suggested_ean","suggested_name","name_sim","visual_sim",
                "confidence","target_cat_id","kassal_url","notes"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(csv_rows)

    html = build_html(sections, len(no_code_cats), len(discontinued_ids))
    html_path.write_text(html, encoding="utf-8")

    print(f"\nCSV  → {csv_path}")
    print(f"HTML → {html_path}")
    print(f"open '{html_path}'")


if __name__ == "__main__":
    main()
