"""
Blur Detection for Reference Product Images
--------------------------------------------
Scans all reference images in NM_NGD_product_images/, computes a blur score
using Laplacian variance (low = blurry), and outputs an HTML gallery of the
most blurry images with product ID, name, and category info.

Usage:
  python3 data/blur_detect.py
  python3 data/blur_detect.py --threshold 80 --output data/audit_out/blur.html
"""

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Install Pillow: pip install Pillow")

try:
    import numpy as np
except ImportError:
    raise SystemExit("Install numpy: pip install numpy")

# ─── Paths ────────────────────────────────────────────────────────────────────

PRODUCT_IMAGES   = Path.home() / "Downloads" / "NM_NGD_product_images"
CATEGORY_MAPPING = Path(__file__).parent / "category_mapping.json"
OUT_DIR          = Path(__file__).parent / "audit_out"

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = 500.0   # Laplacian variance below this = blurry
THUMB_MAX_DIM     = 220
THUMB_QUALITY     = 80

IMAGE_TYPES = ["front", "main", "back", "left", "right", "top"]

# ─── Blur detection ───────────────────────────────────────────────────────────

def laplacian_variance(img: Image.Image) -> float:
    """Compute Laplacian variance as blur score. Lower = more blurry."""
    gray = np.array(img.convert("L"), dtype=np.float32)
    # Laplacian via finite differences (no external deps)
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1]
           + gray[1:-1, :-2] + gray[1:-1, 2:]
           - 4.0 * gray[1:-1, 1:-1])
    return float(np.var(lap))


def thumb_b64(img: Image.Image, max_dim: int = THUMB_MAX_DIM) -> str:
    w, h = img.size
    scale = max_dim / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=THUMB_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()


# ─── Build reverse map: product_code → [category_ids] + product_name ─────────

def load_product_map():
    """Returns dict: product_code → {name, category_ids: [int]}"""
    if not CATEGORY_MAPPING.exists():
        return {}
    raw = json.load(open(CATEGORY_MAPPING))
    result = {}
    for cat_id_str, info in raw.items():
        code = info.get("product_code")
        if not code:
            continue
        if code not in result:
            result[code] = {
                "name": info.get("product_name", ""),
                "category_ids": [],
            }
        result[code]["category_ids"].append(int(cat_id_str))
    return result


# ─── Scan images ──────────────────────────────────────────────────────────────

def scan_images(threshold: float):
    """
    Scan all reference images and return list of blurry findings.
    Each finding: {product_code, product_name, category_ids, image_type, path, score}
    Sorted by score ascending (worst first).
    """
    if not PRODUCT_IMAGES.exists():
        raise SystemExit(f"Product images directory not found: {PRODUCT_IMAGES}")

    product_map = load_product_map()
    findings = []

    product_dirs = sorted(PRODUCT_IMAGES.iterdir())
    total = len(product_dirs)
    print(f"Scanning {total} product folders in {PRODUCT_IMAGES} …")

    for i, folder in enumerate(product_dirs):
        if not folder.is_dir():
            continue
        if i % 100 == 0 and i > 0:
            print(f"  {i}/{total} …")

        product_code = folder.name
        info = product_map.get(product_code, {})
        product_name = info.get("name", "")
        category_ids = info.get("category_ids", [])

        for img_path in sorted(folder.glob("*.jpg")):
            img_type = img_path.stem  # front, main, back, etc.
            try:
                img = Image.open(img_path)
                score = laplacian_variance(img)
            except Exception:
                continue

            if score < threshold:
                findings.append({
                    "product_code": product_code,
                    "product_name": product_name,
                    "category_ids": category_ids,
                    "image_type": img_type,
                    "path": img_path,
                    "score": score,
                })

    findings.sort(key=lambda x: x["score"])
    return findings


# ─── HTML builder ─────────────────────────────────────────────────────────────

def blur_badge(score: float) -> str:
    if score < 50:
        return '<span class="badge red">VELDIG UKLAR</span>'
    if score < 200:
        return '<span class="badge orange">UKLAR</span>'
    return '<span class="badge yellow">LITT UKLAR</span>'


def build_html(findings: list, threshold: float) -> str:
    cards = []
    for f in findings:
        try:
            img = Image.open(f["path"])
            b64 = thumb_b64(img)
        except Exception:
            continue

        cat_str = ", ".join(f"#{c}" for c in sorted(f["category_ids"])) if f["category_ids"] else "—"
        badge = blur_badge(f["score"])
        bar_width = min(100, int(f["score"] / threshold * 100))
        bar_color = "#c0392b" if f["score"] < 20 else ("#e67e22" if f["score"] < 50 else "#f39c12")

        cards.append(f"""
<div class="card">
  <img src="data:image/jpeg;base64,{b64}" title="{f['path']}">
  <div class="card-body">
    <div class="card-name">{f['product_name'] or '—'}</div>
    <div class="card-meta">
      <span class="label">EAN:</span> {f['product_code']}<br>
      <span class="label">Billedtype:</span> {f['image_type']}<br>
      <span class="label">Kategori-ID:</span> {cat_str}<br>
      <span class="label">Blur score:</span> {f['score']:.1f} (threshold: {threshold:.0f})
    </div>
    {badge}
    <div class="score-bar-bg">
      <div class="score-bar" style="width:{bar_width}%;background:{bar_color}"></div>
    </div>
  </div>
</div>""")

    total_blurry = len(findings)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Uklare referansebilder</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: Arial, sans-serif;
    background: #1a1a2e;
    color: #eee;
    margin: 0;
    padding: 20px;
  }}
  h1 {{ color: #e94560; margin-bottom: 4px; }}
  .summary {{
    background: #16213e;
    border-left: 4px solid #e94560;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 20px;
    font-size: 0.9em;
    color: #ccc;
  }}
  .summary strong {{ color: #e94560; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 16px;
  }}
  .card {{
    background: #16213e;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #0f3460;
    display: flex;
    flex-direction: column;
  }}
  .card img {{
    width: 100%;
    height: 200px;
    object-fit: contain;
    background: #0f3460;
    display: block;
  }}
  .card-body {{
    padding: 10px 12px 12px;
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .card-name {{
    font-size: 0.82em;
    font-weight: bold;
    color: #e94560;
    line-height: 1.3;
    min-height: 2.6em;
  }}
  .card-meta {{
    font-size: 0.72em;
    color: #aaa;
    line-height: 1.7;
  }}
  .label {{ color: #888; }}
  .badge {{
    display: inline-block;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72em;
    font-weight: bold;
    margin-top: 2px;
  }}
  .red    {{ background: #c0392b; color: #fff; }}
  .orange {{ background: #e67e22; color: #fff; }}
  .yellow {{ background: #f39c12; color: #000; }}
  .score-bar-bg {{
    background: #0f3460;
    border-radius: 3px;
    height: 5px;
    margin-top: 6px;
    overflow: hidden;
  }}
  .score-bar {{
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s;
  }}
  .filter-bar {{
    margin-bottom: 16px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    align-items: center;
  }}
  .filter-btn {{
    background: #0f3460;
    border: none;
    color: #eee;
    padding: 6px 14px;
    border-radius: 20px;
    cursor: pointer;
    font-size: 0.82em;
  }}
  .filter-btn.active {{ background: #e94560; }}
  .count-badge {{
    background: #0f3460;
    border-radius: 10px;
    padding: 2px 8px;
    font-size: 0.75em;
    color: #aaa;
    margin-left: 4px;
  }}
  input[type=text] {{
    background: #0f3460;
    border: 1px solid #555;
    color: #eee;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.82em;
    outline: none;
    min-width: 200px;
  }}
</style>
</head>
<body>
<h1>Uklare referansebilder</h1>
<div class="summary">
  <strong>{total_blurry}</strong> bilder med blur score under <strong>{threshold:.0f}</strong> — sortert etter skarphet (verst først).<br>
  Blur score = Laplacian-varianse. Lavere = mer uskarp. Threshold: {threshold:.0f}.
</div>

<div class="filter-bar">
  <input type="text" id="search" placeholder="Søk på produkt eller EAN…" oninput="filterCards()">
  <button class="filter-btn active" onclick="setFilter('all', this)">
    Alle <span class="count-badge" id="cnt-all">{total_blurry}</span>
  </button>
  <button class="filter-btn" onclick="setFilter('very', this)">
    Veldig uklar <span class="count-badge" id="cnt-very"></span>
  </button>
  <button class="filter-btn" onclick="setFilter('blurry', this)">
    Uklar <span class="count-badge" id="cnt-blurry"></span>
  </button>
  <button class="filter-btn" onclick="setFilter('slight', this)">
    Litt uklar <span class="count-badge" id="cnt-slight"></span>
  </button>
</div>

<div class="grid" id="grid">
{"".join(cards)}
</div>

<script>
let activeFilter = 'all';

function getLevel(card) {{
  const scoreText = card.querySelector('.card-meta').innerText;
  const m = scoreText.match(/Blur score:\\s*([\\d.]+)/);
  if (!m) return 'slight';
  const s = parseFloat(m[1]);
  if (s < 20) return 'very';
  if (s < 50) return 'blurry';
  return 'slight';
}}

function filterCards() {{
  const q = document.getElementById('search').value.toLowerCase();
  const cards = document.querySelectorAll('.card');
  let counts = {{all: 0, very: 0, blurry: 0, slight: 0}};

  cards.forEach(c => {{
    const text = c.innerText.toLowerCase();
    const level = getLevel(c);
    const matchSearch = !q || text.includes(q);
    const matchFilter = activeFilter === 'all' || level === activeFilter;
    const show = matchSearch && matchFilter;
    c.style.display = show ? '' : 'none';
    if (matchSearch) {{
      counts.all++;
      counts[level]++;
    }}
  }});

  document.getElementById('cnt-all').textContent = counts.all;
  document.getElementById('cnt-very').textContent = counts.very;
  document.getElementById('cnt-blurry').textContent = counts.blurry;
  document.getElementById('cnt-slight').textContent = counts.slight;
}}

function setFilter(f, btn) {{
  activeFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  filterCards();
}}

// Initialize counts
window.onload = filterCards;
</script>
</body>
</html>"""


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Detect blurry reference product images")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Laplacian variance threshold (default {DEFAULT_THRESHOLD}). Lower = stricter.")
    parser.add_argument("--output", default=None, help="Output HTML path")
    args = parser.parse_args()

    findings = scan_images(args.threshold)
    print(f"\nFunnet {len(findings)} uklare bilder under threshold {args.threshold:.0f}")

    if not findings:
        print("Ingen uklare bilder funnet. Prøv høyere --threshold.")
        return

    # Print summary
    very   = sum(1 for f in findings if f["score"] < 50)
    blurry = sum(1 for f in findings if 50 <= f["score"] < 200)
    slight = sum(1 for f in findings if f["score"] >= 200)
    print(f"  Veldig uklar (< 50):    {very}")
    print(f"  Uklar (50–200):         {blurry}")
    print(f"  Litt uklar (200–{args.threshold:.0f}): {slight}")
    print()

    html = build_html(findings, args.threshold)

    out = Path(args.output) if args.output else OUT_DIR / "blur.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Galleri lagret → {out}")
    print(f"\nÅpne i nettleser:")
    print(f"  open '{out}'")


if __name__ == "__main__":
    main()
