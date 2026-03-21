"""
Reclassify cat#355 (unknown_product) annotations.

For each annotation in cat#355, compares its perceptual hash against the
centroid hash of every other category and finds the best match.
Outputs an HTML gallery sorted by match confidence so you can review and
approve/reject each suggestion before patching.

Algorithm:
  1. Build centroid hash for each category (from existing hash_cache.json)
  2. For each cat#355 annotation, find category with lowest Hamming distance
  3. Rank by confidence (lower distance = more confident match)
  4. Output HTML for human review

Usage:
  python3 data/reclassify_unknown.py
  python3 data/reclassify_unknown.py --output data/audit_out/reclassify.html
"""

import argparse
import base64
import json
from collections import defaultdict
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
HASH_CACHE_PATH  = Path(__file__).parent / "audit_out" / "hash_cache.json"
KASSAL_IMAGES    = Path(__file__).parent / "audit_out" / "kassal_images"
OUT_DIR          = Path(__file__).parent / "audit_out"

UNKNOWN_CAT_ID = 355
HASH_SIZE      = 16
CROP_PADDING   = 6
THUMB_DIM      = 160
REF_DIM        = 140
MIN_CROP_PX    = 20

# ─── Hash helpers (same as audit_outliers.py) ─────────────────────────────────

def hamming(h1, h2):
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

def thumb_b64(img, max_dim):
    if img is None:
        return None
    w, h = img.size
    scale = max_dim / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=75)
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

# ─── Confidence badge ─────────────────────────────────────────────────────────

def confidence(dist):
    """Convert Hamming distance to confidence label."""
    if dist < 0.20:
        return "STERK", "#27ae60"
    if dist < 0.30:
        return "MIDDELS", "#f39c12"
    if dist < 0.40:
        return "SVAK", "#e67e22"
    return "INGEN MATCH", "#c0392b"

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--top", type=int, default=3,
                        help="Number of top candidate categories to show per annotation (default 3)")
    args = parser.parse_args()

    print("Laster annotations og hash-cache …")
    data       = json.load(open(ANNOTATIONS_PATH))
    cats       = {c["id"]: c["name"] for c in data["categories"]}
    img_map    = {i["id"]: i["file_name"] for i in data["images"]}
    mapping    = {}
    if CATEGORY_MAPPING.exists():
        mapping = {int(k): v for k, v in json.load(open(CATEGORY_MAPPING)).items()}

    # Load hash cache
    raw_cache = json.load(open(HASH_CACHE_PATH))
    hash_cache = {}
    for k, v in raw_cache.items():
        if isinstance(v, str):
            hash_cache[int(k)] = int(v) if v else None
        elif isinstance(v, int):
            hash_cache[int(k)] = v
        else:
            hash_cache[int(k)] = v  # None

    # Build centroid per category (all except 355)
    print("Beregner sentroid-hash per kategori …")
    ann_by_cat = defaultdict(list)
    for a in data["annotations"]:
        ann_by_cat[a["category_id"]].append(a)

    cat_centroids = {}
    for cid, anns in ann_by_cat.items():
        if cid == UNKNOWN_CAT_ID:
            continue
        hashes = [hash_cache[a["id"]] for a in anns
                  if hash_cache.get(a["id"]) is not None]
        if len(hashes) < 2:
            continue
        cat_centroids[cid] = centroid_hash(hashes)

    print(f"Kategorier med sentroid: {len(cat_centroids)}")

    # For each cat#355 annotation, find best matches
    unknown_anns = ann_by_cat[UNKNOWN_CAT_ID]
    print(f"Matcher {len(unknown_anns)} ukjente annotations …")

    results = []
    for ann in unknown_anns:
        ann_hash = hash_cache.get(ann["id"])
        if ann_hash is None:
            continue

        # Score against all centroids
        scored = []
        for cid, centroid in cat_centroids.items():
            dist = hamming(ann_hash, centroid)
            scored.append((dist, cid))
        scored.sort()

        best_dist, best_cat = scored[0]
        top_candidates = scored[:args.top]

        results.append({
            "ann":        ann,
            "best_dist":  best_dist,
            "best_cat":   best_cat,
            "candidates": top_candidates,
        })

    # Sort: best matches first (lowest distance)
    results.sort(key=lambda x: x["best_dist"])

    print(f"Bygger HTML for {len(results)} annotations …")

    # ─── Stats ────────────────────────────────────────────────────────────────
    strong  = sum(1 for r in results if r["best_dist"] < 0.20)
    medium  = sum(1 for r in results if 0.20 <= r["best_dist"] < 0.30)
    weak    = sum(1 for r in results if 0.30 <= r["best_dist"] < 0.40)
    no_match = sum(1 for r in results if r["best_dist"] >= 0.40)

    # ─── Build HTML cards ─────────────────────────────────────────────────────
    cards = []
    for r in results:
        ann       = r["ann"]
        ann_id    = ann["id"]
        img_id    = ann["image_id"]
        bbox      = ann["bbox"]

        # Shelf crop
        shelf_file = img_map.get(img_id)
        crop_img   = load_crop(IMAGES_DIR / shelf_file, bbox) if shelf_file else None
        crop_b64   = thumb_b64(crop_img, THUMB_DIM)
        if crop_b64 is None:
            continue

        conf_label, conf_color = confidence(r["best_dist"])

        # Candidate panels
        cand_panels = []
        for rank, (dist, cid) in enumerate(r["candidates"]):
            cname = cats.get(cid, f"cat_{cid}")
            info  = mapping.get(cid, {})
            code  = info.get("product_code")

            ref_path = best_ref(code)
            ref_img  = None
            if ref_path:
                try:
                    ref_img = Image.open(ref_path)
                except Exception:
                    pass
            kas_path = kassal_img(code)
            kas_img_obj = None
            if kas_path:
                try:
                    kas_img_obj = Image.open(kas_path)
                except Exception:
                    pass

            ref_b64 = thumb_b64(ref_img, REF_DIM)
            kas_b64 = thumb_b64(kas_img_obj, REF_DIM)

            ref_html = (f'<img src="data:image/jpeg;base64,{ref_b64}" class="ref-img" title="Referansebilde">'
                        if ref_b64 else '<div class="no-img">ingen ref</div>')
            kas_html = (f'<img src="data:image/jpeg;base64,{kas_b64}" class="ref-img" title="Kassal">'
                        if kas_b64 else '<div class="no-img">ingen kassal</div>')

            cl_dist, cl_color = confidence(dist)
            border = "2px solid #27ae60" if rank == 0 else "1px solid #0f3460"
            cand_panels.append(f"""
<div class="candidate" style="border:{border}">
  <div class="cand-rank">#{rank+1} — dist={dist:.3f}</div>
  <div class="cand-name" title="cat#{cid}">cat#{cid} {cname[:35]}</div>
  <div class="cand-imgs">
    {ref_html}
    {kas_html}
  </div>
</div>""")

        bx, by, bw, bh = [int(v) for v in bbox]
        cards.append(f"""
<div class="card" data-conf="{conf_label.lower().replace(' ','-')}" data-dist="{r['best_dist']:.3f}">
  <div class="card-header" style="border-left:4px solid {conf_color}">
    <div class="ann-info">
      ann#{ann_id} &nbsp;|&nbsp; img#{img_id} &nbsp;|&nbsp; bbox [{bx},{by},{bw},{bh}]
    </div>
    <span class="badge" style="background:{conf_color}">{conf_label}</span>
  </div>
  <div class="card-body">
    <div class="crop-panel">
      <div class="panel-lbl">Ukjent crop</div>
      <img src="data:image/jpeg;base64,{crop_b64}" class="crop-img">
    </div>
    <div class="candidates-row">
      <div class="panel-lbl">Beste kandidater (lavest Hamming-avstand fra kategori-sentroid)</div>
      <div class="cands">{"".join(cand_panels)}</div>
    </div>
  </div>
</div>""")

    # ─── Assemble HTML ────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Reklassifisering av unknown_product (cat#355)</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: Arial, sans-serif; background: #1a1a2e; color: #eee;
         margin: 0; padding: 20px; }}
  h1 {{ color: #e94560; margin-bottom: 4px; }}
  .summary {{ background: #16213e; border-left: 4px solid #e94560; border-radius:6px;
              padding: 12px 16px; margin-bottom: 16px; font-size:.88em; color:#ccc; }}
  .summary strong {{ color:#e94560; }}
  .stat-row {{ display:flex; gap:20px; margin-top:8px; }}
  .stat {{ background:#0f3460; border-radius:6px; padding:6px 14px; font-size:.85em; }}
  .filter-bar {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px; align-items:center; }}
  .fbtn {{ background:#0f3460; border:none; color:#eee; padding:6px 14px;
           border-radius:20px; cursor:pointer; font-size:.82em; }}
  .fbtn.active {{ background:#e94560; }}
  .cnt {{ background:#0f3460; border-radius:10px; padding:2px 7px;
          font-size:.72em; color:#aaa; margin-left:3px; }}
  .card {{ background:#16213e; border-radius:8px; margin:12px 0;
           overflow:hidden; }}
  .card-header {{ display:flex; justify-content:space-between; align-items:center;
                  padding:8px 14px; background:#0d1b33; }}
  .ann-info {{ font-size:.78em; color:#aaa; font-family:monospace; }}
  .badge {{ border-radius:4px; padding:3px 10px; font-size:.75em; font-weight:bold; color:#fff; }}
  .card-body {{ display:flex; gap:14px; padding:12px 14px; align-items:flex-start;
                flex-wrap:wrap; }}
  .crop-panel {{ text-align:center; min-width:180px; }}
  .crop-img {{ max-width:{THUMB_DIM}px; max-height:{THUMB_DIM}px; border-radius:4px;
               border:2px solid #0f3460; display:block; margin:4px auto; }}
  .panel-lbl {{ font-size:.72em; color:#888; margin-bottom:6px; }}
  .candidates-row {{ flex:1; }}
  .cands {{ display:flex; gap:10px; flex-wrap:wrap; }}
  .candidate {{ background:#0f3460; border-radius:6px; padding:8px; min-width:150px;
                max-width:200px; }}
  .cand-rank {{ font-size:.68em; color:#aaa; font-family:monospace; }}
  .cand-name {{ font-size:.75em; color:#e94560; margin:3px 0 6px; font-weight:bold;
                word-break:break-word; line-height:1.3; }}
  .cand-imgs {{ display:flex; gap:6px; flex-wrap:wrap; }}
  .ref-img {{ max-width:{REF_DIM}px; max-height:{REF_DIM}px; border-radius:3px;
              border:1px solid #1a1a2e; display:block; }}
  .no-img {{ width:60px; height:60px; background:#1a1a2e; border-radius:4px;
             display:flex; align-items:center; justify-content:center;
             font-size:.65em; color:#666; text-align:center; }}
</style>
</head>
<body>
<h1>Reklassifisering av unknown_product (cat#355)</h1>
<div class="summary">
  <strong>{len(results)}</strong> annotations i cat#355 matchet mot {len(cat_centroids)} kategori-sentrioider.<br>
  Hvert kort viser den ukjente crop-en og de <strong>{args.top} beste kandidatkategoriene</strong> (sortert etter Hamming-avstand).<br>
  <strong>Grønn kant = beste kandidat.</strong> Lavere dist = sikrere match.
  <div class="stat-row">
    <div class="stat" style="border-left:3px solid #27ae60">Sterk match (&lt;0.20): <strong>{strong}</strong></div>
    <div class="stat" style="border-left:3px solid #f39c12">Middels (0.20–0.30): <strong>{medium}</strong></div>
    <div class="stat" style="border-left:3px solid #e67e22">Svak (0.30–0.40): <strong>{weak}</strong></div>
    <div class="stat" style="border-left:3px solid #c0392b">Ingen match (&gt;0.40): <strong>{no_match}</strong></div>
  </div>
</div>

<div class="filter-bar">
  <button class="fbtn active" onclick="setFilter('all',this)">
    Alle <span class="cnt" id="cnt-all">{len(results)}</span>
  </button>
  <button class="fbtn" onclick="setFilter('sterk',this)">
    Sterk match <span class="cnt" id="cnt-sterk">{strong}</span>
  </button>
  <button class="fbtn" onclick="setFilter('middels',this)">
    Middels <span class="cnt" id="cnt-middels">{medium}</span>
  </button>
  <button class="fbtn" onclick="setFilter('svak',this)">
    Svak <span class="cnt" id="cnt-svak">{weak}</span>
  </button>
  <button class="fbtn" onclick="setFilter('ingen-match',this)">
    Ingen match <span class="cnt" id="cnt-ingen-match">{no_match}</span>
  </button>
</div>

<div id="cards-container">
{"".join(cards)}
</div>

<script>
let activeFilter = 'all';
function setFilter(f, btn) {{
  activeFilter = f;
  document.querySelectorAll('.fbtn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.card').forEach(c => {{
    const show = f === 'all' || c.dataset.conf === f;
    c.style.display = show ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""

    out = Path(args.output) if args.output else OUT_DIR / "reclassify.html"
    out.write_text(html, encoding="utf-8")
    print(f"\nSummary:")
    print(f"  Sterk match (<0.20):    {strong}")
    print(f"  Middels (0.20–0.30):    {medium}")
    print(f"  Svak (0.30–0.40):       {weak}")
    print(f"  Ingen match (>0.40):    {no_match}")
    print(f"\nLagret → {out}")
    print(f"open '{out}'")


if __name__ == "__main__":
    main()
