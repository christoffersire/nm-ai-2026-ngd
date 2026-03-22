"""
Viser produkter som har bilder i NM_NGD_product_images/ men ingen kategori
i annotations.json — potensielt manglende annotations.
"""

import base64
import json
from io import BytesIO
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("pip install Pillow")

PRODUCT_IMAGES   = Path.home() / "Downloads" / "NM_NGD_product_images"
ANNOTATIONS_PATH = Path.home() / "Downloads" / "train" / "annotations.json"
CATEGORY_MAPPING = Path(__file__).parent / "category_mapping.json"
OUT_DIR          = Path(__file__).parent / "audit_out"
THUMB_DIM        = 200
REF_PRIORITY     = ["front.jpg", "main.jpg", "back.jpg", "left.jpg", "right.jpg", "top.jpg"]


def thumb_b64(path, max_dim=THUMB_DIM):
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = max_dim / max(w, h)
        if scale < 1:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def main():
    data    = json.load(open(ANNOTATIONS_PATH))
    mapping = json.load(open(CATEGORY_MAPPING)) if CATEGORY_MAPPING.exists() else {}

    annotation_eans = {v.get("product_code") for v in mapping.values() if v.get("product_code")}
    image_eans      = {f.name for f in PRODUCT_IMAGES.iterdir() if f.is_dir()}
    missing         = sorted(image_eans - annotation_eans)

    # Also flag the one with annotations but no images
    only_in_anns = sorted(annotation_eans - image_eans)

    print(f"Produkter med bilder men ingen annotation-kategori: {len(missing)}")

    cards = []
    for ean in missing:
        folder = PRODUCT_IMAGES / ean
        imgs   = []
        for name in REF_PRIORITY:
            p = folder / name
            if p.exists():
                imgs.append(p)
        for p in sorted(folder.glob("*.jpg")):
            if p not in imgs:
                imgs.append(p)

        is_custom = ean.startswith("CUSTOM_")

        img_tags = []
        for p in imgs:
            b64 = thumb_b64(p)
            if b64:
                img_tags.append(
                    f'<div class="img-wrap">'
                    f'<img src="data:image/jpeg;base64,{b64}">'
                    f'<div class="lbl">{p.stem}</div>'
                    f'</div>'
                )

        tag_class = "card custom" if is_custom else "card real"
        badge = '<span class="badge custom-badge">CUSTOM</span>' if is_custom else '<span class="badge ean-badge">EAN</span>'

        cards.append(f"""
<div class="{tag_class}">
  <div class="card-header">
    <span class="ean-code">{ean}</span> {badge}
    <span class="img-count">{len(imgs)} bilde{'r' if len(imgs)!=1 else ''}</span>
  </div>
  <div class="img-row">
    {"".join(img_tags) if img_tags else '<span class="no-img">ingen bilder</span>'}
  </div>
  <div class="card-footer">
    Finnes <strong>ikke</strong> som kategori i annotations.json
  </div>
</div>""")

    # Card for AXA Müsli (has annotations, now has images)
    fixed_cards = []
    for ean in only_in_anns:
        folder = PRODUCT_IMAGES / ean
        cats_using = [(k, v.get("product_name","")) for k, v in mapping.items()
                      if v.get("product_code") == ean]
        imgs = list(folder.glob("*.jpg")) if folder.exists() else []
        img_tags = []
        for p in imgs:
            b64 = thumb_b64(p)
            if b64:
                img_tags.append(f'<div class="img-wrap"><img src="data:image/jpeg;base64,{b64}"><div class="lbl">{p.stem}</div></div>')
        status = "✓ Bildemappen nettopp opprettet" if folder.exists() else "✗ Ingen bildemapper"
        for cid, name in cats_using:
            fixed_cards.append(f"""
<div class="card fixed">
  <div class="card-header">
    <span class="ean-code">{ean}</span>
    <span class="badge fixed-badge">FIKSET</span>
    <span class="img-count">cat#{cid} — {name}</span>
  </div>
  <div class="img-row">{"".join(img_tags) if img_tags else '<span class="no-img">ingen bilder ennå</span>'}</div>
  <div class="card-footer">{status}</div>
</div>""")

    n_real   = sum(1 for e in missing if not e.startswith("CUSTOM_"))
    n_custom = sum(1 for e in missing if e.startswith("CUSTOM_"))

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Produkter uten annotations</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: Arial, sans-serif; background: #1a1a2e; color: #eee;
         margin: 0; padding: 20px; }}
  h1 {{ color: #e94560; margin-bottom: 4px; }}
  h2 {{ color: #e94560; margin: 24px 0 10px; font-size: 1em; text-transform: uppercase;
        letter-spacing: 1px; }}
  .summary {{ background: #16213e; border-left: 4px solid #e94560; border-radius: 6px;
              padding: 12px 16px; margin-bottom: 20px; font-size: .88em; color: #ccc; }}
  .summary strong {{ color: #e94560; }}
  .stats {{ display: flex; gap: 14px; margin-top: 8px; flex-wrap: wrap; }}
  .stat {{ background: #0f3460; border-radius: 6px; padding: 6px 14px; font-size: .82em; }}
  .filter-bar {{ display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; }}
  .fbtn {{ background: #0f3460; border: none; color: #eee; padding: 6px 14px;
           border-radius: 20px; cursor: pointer; font-size: .82em; }}
  .fbtn.active {{ background: #e94560; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
           gap: 16px; }}
  .card {{ background: #16213e; border-radius: 8px; overflow: hidden;
           border: 1px solid #0f3460; }}
  .card.custom {{ border-color: #4a3060; }}
  .card.fixed  {{ border-color: #27ae60; }}
  .card-header {{ display: flex; align-items: center; gap: 8px; padding: 10px 12px;
                  background: #0d1b33; flex-wrap: wrap; }}
  .ean-code {{ font-family: monospace; font-size: .85em; color: #e94560; font-weight: bold; }}
  .img-count {{ font-size: .72em; color: #888; margin-left: auto; }}
  .img-row {{ display: flex; flex-wrap: wrap; gap: 6px; padding: 10px 12px; min-height: 80px; }}
  .img-wrap {{ text-align: center; }}
  .img-wrap img {{ max-width: {THUMB_DIM}px; max-height: {THUMB_DIM}px;
                   border-radius: 4px; display: block; }}
  .lbl {{ font-size: .65em; color: #888; margin-top: 2px; }}
  .no-img {{ font-size: .8em; color: #555; font-style: italic; padding: 8px; }}
  .card-footer {{ font-size: .75em; color: #aaa; padding: 8px 12px;
                  border-top: 1px solid #0f3460; background: #111827; }}
  .badge {{ border-radius: 4px; padding: 2px 8px; font-size: .7em; font-weight: bold; }}
  .ean-badge    {{ background: #0f3460; color: #7ec8e3; }}
  .custom-badge {{ background: #4a3060; color: #c88de3; }}
  .fixed-badge  {{ background: #27ae60; color: #fff; }}
</style>
</head>
<body>
<h1>Produkter uten annotations</h1>
<div class="summary">
  Disse produktene har bilder i <code>NM_NGD_product_images/</code> men finnes ikke som kategori i <code>annotations.json</code>.<br>
  De kan være produkter som faktisk er på hyllene men aldri ble annotert — eller ekstra produkter lagt inn av oppgavegiverne.
  <div class="stats">
    <div class="stat" style="border-left:3px solid #7ec8e3">Ekte EAN-produkter: <strong>{n_real}</strong></div>
    <div class="stat" style="border-left:3px solid #c88de3">CUSTOM (ingen EAN): <strong>{n_custom}</strong></div>
    <div class="stat" style="border-left:3px solid #27ae60">Fikset (bilder lagt til): <strong>{len(fixed_cards)}</strong></div>
  </div>
</div>

<div class="filter-bar">
  <button class="fbtn active" onclick="setFilter('all',this)">Alle</button>
  <button class="fbtn" onclick="setFilter('real',this)">Ekte EAN ({n_real})</button>
  <button class="fbtn" onclick="setFilter('custom',this)">CUSTOM ({n_custom})</button>
  <button class="fbtn" onclick="setFilter('fixed',this)">Fikset ({len(fixed_cards)})</button>
</div>

<div class="grid" id="grid">
{"".join(cards)}
{"".join(fixed_cards)}
</div>

<script>
function setFilter(f, btn) {{
  document.querySelectorAll('.fbtn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.card').forEach(c => {{
    c.style.display = (f === 'all' || c.classList.contains(f)) ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""

    out = OUT_DIR / "missing_annotations.html"
    out.write_text(html, encoding="utf-8")
    print(f"Lagret → {out}  ({len(missing)} produkter + {len(fixed_cards)} fikset)")
    print(f"open '{out}'")


if __name__ == "__main__":
    main()
