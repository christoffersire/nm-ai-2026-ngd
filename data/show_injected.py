"""
Viser de injiserte referansebildene per kategori i et HTML-galleri
så du kan dobbelsjekke at riktig produkt ble koblet til riktig kategori.

Usage:
  python3 data/show_injected.py
"""

import base64
import json
from io import BytesIO
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("pip install Pillow")

ANNOTATIONS_PATH = Path.home() / "Downloads" / "train" / "annotations.json"
IMAGES_DIR       = Path.home() / "Downloads" / "train" / "images"
PRODUCT_IMAGES   = Path.home() / "Downloads" / "NM_NGD_product_images"
CATEGORY_MAPPING = Path(__file__).parent / "category_mapping.json"
CLASSIFIER_DIR   = Path(__file__).parent.parent / "datasets" / "classifier"
OUT_DIR          = Path(__file__).parent / "audit_out"

THUMB_DIM  = 140
CROP_DIM   = 120
REF_PRIORITY = ["front.jpg", "main.jpg", "back.jpg", "left.jpg", "right.jpg", "top.jpg"]


def thumb_b64(img, max_dim):
    if img is None:
        return None
    w, h = img.size
    scale = max_dim / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    data    = json.load(open(ANNOTATIONS_PATH))
    cats    = {c["id"]: c["name"] for c in data["categories"]}
    img_map = {i["id"]: i["file_name"] for i in data["images"]}
    mapping = {}
    if CATEGORY_MAPPING.exists():
        mapping = {int(k): v for k, v in json.load(open(CATEGORY_MAPPING)).items()}

    from collections import defaultdict
    ann_by_cat = defaultdict(list)
    for a in data["annotations"]:
        ann_by_cat[a["category_id"]].append(a)

    train_dir = CLASSIFIER_DIR / "train"

    # Find categories that have injected ref images
    sections = []
    for cat_folder in sorted(train_dir.iterdir()):
        if not cat_folder.is_dir():
            continue
        cat_id = int(cat_folder.name)

        ref_files = sorted(cat_folder.glob("aug_*.jpg"))
        if not ref_files:
            continue  # no augmented images

        name   = cats.get(cat_id, f"cat_{cat_id}")
        info   = mapping.get(cat_id, {})
        code   = info.get("product_code", "")
        gt_files = [f for f in cat_folder.glob("crop_img*.jpg") if "_jit" not in f.name]

        # Show a sample of the augmented images (every 5th to avoid too many)
        aug_sample = ref_files[::5][:8]
        source_cards = []
        for p in aug_sample:
            try:
                img = Image.open(p)
                b64 = thumb_b64(img, THUMB_DIM)
            except Exception:
                continue
            source_cards.append(
                f'<div class="img-card">'
                f'<img src="data:image/jpeg;base64,{b64}">'
                f'<div class="lbl">{p.stem[-10:]}</div>'
                f'</div>'
            )

        # Build a few GT shelf crop examples for comparison
        gt_cards = []
        import random
        random.seed(42)
        sample_anns = random.sample(ann_by_cat[cat_id], min(4, len(ann_by_cat[cat_id])))
        for ann in sample_anns:
            shelf_file = img_map.get(ann["image_id"])
            if not shelf_file:
                continue
            shelf_path = IMAGES_DIR / shelf_file
            try:
                shelf = Image.open(shelf_path)
                x, y, w, h = [int(v) for v in ann["bbox"]]
                iw, ih = shelf.size
                crop = shelf.crop((max(0,x-6), max(0,y-6), min(iw,x+w+6), min(ih,y+h+6)))
                b64 = thumb_b64(crop, CROP_DIM)
            except Exception:
                continue
            gt_cards.append(
                f'<div class="img-card">'
                f'<img src="data:image/jpeg;base64,{b64}">'
                f'<div class="lbl">ann#{ann["id"]}</div>'
                f'</div>'
            )

        sections.append(f"""
<section class="cat-section">
  <div class="cat-header">
    <div class="cat-title">cat#{cat_id} — {name}</div>
    <div class="cat-meta">
      EAN: {code or "—"} &nbsp;|&nbsp;
      GT crops: {len(gt_files)} &nbsp;|&nbsp;
      Augmenterte bilder: {len(ref_files)}
      ({len(gt_files)} GT crops × 25 augments)
    </div>
  </div>
  <div class="panels">
    <div class="panel">
      <div class="panel-lbl">Eksempler på augmenterte GT-crops (utvalg av {len(ref_files)} totalt)</div>
      <div class="img-row">{"".join(source_cards) if source_cards else '<span class="missing">ingen augmenterte bilder</span>'}</div>
    </div>
    <div class="panel">
      <div class="panel-lbl">GT hyllecrops fra treningssett (til sammenligning)</div>
      <div class="img-row">{"".join(gt_cards) if gt_cards else '<span class="missing">ingen GT crops</span>'}</div>
    </div>
  </div>
</section>""")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Injiserte referansebilder — dobbelsjekk</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: Arial, sans-serif; background: #1a1a2e; color: #eee;
         margin: 0; padding: 20px; }}
  h1 {{ color: #e94560; margin-bottom: 4px; }}
  .summary {{ background: #16213e; border-left:4px solid #e94560; border-radius:6px;
              padding:12px 16px; margin-bottom:20px; font-size:.88em; color:#ccc; }}
  .summary strong {{ color:#e94560; }}
  input[type=text] {{ background:#0f3460; border:1px solid #555; color:#eee;
                      padding:7px 14px; border-radius:20px; font-size:.85em;
                      outline:none; min-width:280px; margin-bottom:16px; }}
  .cat-section {{ background:#16213e; border-radius:8px; margin:14px 0;
                  padding:14px 16px; border-left:4px solid #0f3460; }}
  .cat-header {{ margin-bottom:10px; }}
  .cat-title {{ font-size:1.05em; font-weight:bold; color:#e94560; }}
  .cat-meta {{ font-size:.75em; color:#aaa; margin-top:2px; }}
  .panels {{ display:flex; gap:20px; flex-wrap:wrap; }}
  .panel {{ flex:1; min-width:280px; }}
  .panel-lbl {{ font-size:.72em; color:#888; margin-bottom:6px; }}
  .img-row {{ display:flex; flex-wrap:wrap; gap:8px; }}
  .img-card {{ background:#0f3460; border-radius:6px; padding:6px; text-align:center; }}
  .img-card img {{ max-width:{THUMB_DIM}px; max-height:{THUMB_DIM}px;
                   display:block; border-radius:3px; }}
  .lbl {{ font-size:.65em; color:#aaa; margin-top:3px; }}
  .missing {{ font-size:.8em; color:#555; font-style:italic; }}
</style>
</head>
<body>
<h1>Injiserte referansebilder — dobbelsjekk</h1>
<div class="summary">
  Viser <strong>{len(sections)}</strong> kategorier der GT-crops ble augmentert (lavt antall treningsbilder).<br>
  <strong>Venstre panel:</strong> eksempler på augmenterte varianter (flip, rotasjon, lys, crop) av GT-cropsene.<br>
  <strong>Høyre panel:</strong> originale GT hyllecrops (kilden til augmenteringene).<br>
  Begge sider skal vise <em>samme produkt</em> — augmenteringene er laget direkte fra GT-cropsene.
</div>

<input type="text" id="search" placeholder="Søk på produktnavn eller EAN…" oninput="filterSections()">

<div id="container">
{"".join(sections)}
</div>

<script>
function filterSections() {{
  const q = document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('.cat-section').forEach(s => {{
    s.style.display = !q || s.innerText.toLowerCase().includes(q) ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""

    out = OUT_DIR / "injected_refs.html"
    out.write_text(html, encoding="utf-8")
    print(f"Lagret → {out}  ({len(sections)} kategorier)")
    print(f"open '{out}'")


if __name__ == "__main__":
    main()
