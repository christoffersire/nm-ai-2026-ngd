"""
Product validation audit — periodic health check.

Checks the full triangle for every product:
  product_code (EAN/GTIN) <-> product_name <-> reference_image

Also validates annotation linkage so every finding maps back to specific
annotation IDs that can later be corrected.

External lookups:
  - Open Food Facts  : always on, no auth
  - Kassal.app       : Norwegian grocery stores (MENY/SPAR/KIWI/JOKER/Bunnpris)
                       Token in data/audit_config.json
  - Tradesolution   : requires OAuth2 from post@tradesolution.no

Outputs (all in data/audit_out/):
  findings.csv             one row per (product, flag) — the main review list
  annotations_flagged.csv  one row per annotation in a flagged category
  report.json              full machine-readable results
  report.html              visual gallery for human review
  off_cache.json           cached OFF + Kassal responses (speeds up reruns)
  kassal_images/           cached Kassal product images

Usage:
  python data/validate_products.py                # full run
  python data/validate_products.py --limit 10     # quick smoke test
  python data/validate_products.py --skip-ok      # hide clean products from HTML
"""

import csv
import json
import time
import urllib.request
import urllib.error
from difflib import SequenceMatcher
from pathlib import Path
from collections import defaultdict
import argparse

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

ROOT          = Path(__file__).resolve().parent.parent
META_PATH     = Path.home() / "Downloads" / "NM_NGD_product_images" / "metadata.json"
ANN_PATH      = Path.home() / "Downloads" / "train" / "annotations.json"
REF_DIR       = Path.home() / "Downloads" / "NM_NGD_product_images"
OUT_DIR       = ROOT / "data" / "audit_out"
OFF_CACHE     = OUT_DIR / "off_cache.json"
IMG_CACHE_DIR = OUT_DIR / "off_images"
KSL_IMG_DIR   = OUT_DIR / "kassal_images"
CONFIG_PATH   = ROOT / "data" / "audit_config.json"

OUT_DIR.mkdir(exist_ok=True)
IMG_CACHE_DIR.mkdir(exist_ok=True)
KSL_IMG_DIR.mkdir(exist_ok=True)

# Load config
_cfg = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
KASSAL_TOKEN = _cfg.get("kassal_token") or ""

OFF_API       = "https://world.openfoodfacts.org/api/v2/product/{gtin}.json"
KASSAL_API    = "https://kassal.app/api/v1/products/ean/{gtin}"
USER_AGENT    = "NM-AI-2026-DataAudit/1.0 (educational)"
OFF_RATE_S    = 1.1
KASSAL_RATE_S = 1.1  # 60 req/min limit

# ---------------------------------------------------------------------------
# EAN / GTIN validation
# ---------------------------------------------------------------------------

def _ean13_check(code):
    if len(code) != 13 or not code.isdigit():
        return False
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(code[:12]))
    return (10 - total % 10) % 10 == int(code[-1])

def _ean8_check(code):
    if len(code) != 8 or not code.isdigit():
        return False
    total = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(code[:7]))
    return (10 - total % 10) % 10 == int(code[-1])

def validate_gtin(code):
    """Returns {type, valid, normalized}."""
    if not code.isdigit():
        return {"type": "non-numeric", "valid": False, "normalized": None}
    l = len(code)
    if l == 13:
        return {"type": "EAN-13", "valid": _ean13_check(code), "normalized": code}
    if l == 12:
        norm = "0" + code
        return {"type": "UPC-A", "valid": _ean13_check(norm), "normalized": norm}
    if l == 8:
        return {"type": "EAN-8", "valid": _ean8_check(code), "normalized": code}
    if l == 11:
        norm = code.zfill(13)
        valid = _ean13_check(norm)
        return {"type": "EAN-11(truncated)", "valid": valid, "normalized": norm if valid else None}
    return {"type": f"len{l}-unknown", "valid": False, "normalized": None}

# ---------------------------------------------------------------------------
# Name similarity
# ---------------------------------------------------------------------------

def _norm(s):
    return " ".join(
        s.lower()
         .replace("æ","ae").replace("ø","o").replace("å","a")
         .replace("&"," og ").split()
    )

def name_similarity(a, b):
    return round(SequenceMatcher(None, _norm(a), _norm(b)).ratio(), 3)

# ---------------------------------------------------------------------------
# Open Food Facts
# ---------------------------------------------------------------------------

def load_off_cache():
    if OFF_CACHE.exists():
        with open(OFF_CACHE) as f:
            return json.load(f)
    return {}

def save_off_cache(cache):
    OFF_CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))

def query_off(gtin, cache):
    if gtin in cache:
        return cache[gtin]
    url = OFF_API.format(gtin=gtin)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())
        result = data.get("product", {}) if data.get("status") == 1 else {}
    except urllib.error.HTTPError as e:
        result = {"_error": f"HTTP {e.code}"}
    except Exception as e:
        result = {"_error": str(e)}
    cache[gtin] = result
    return result

def off_name(off):
    return (off.get("product_name") or off.get("product_name_en")
            or off.get("product_name_no") or None)

def off_img_url(off):
    return off.get("image_front_url") or off.get("image_url") or None

# ---------------------------------------------------------------------------
# Tradesolution (pluggable — add credentials to enable)
# ---------------------------------------------------------------------------
# To enable:
#   1. Email post@tradesolution.no for a client_id and client_secret
#   2. Set TS_CLIENT_ID and TS_CLIENT_SECRET env vars (or hardcode below)
#   3. Uncomment and fill TS_CLIENT_ID / TS_CLIENT_SECRET

TS_CLIENT_ID     = None  # "your-client-id"
TS_CLIENT_SECRET = None  # "your-client-secret"
TS_TOKEN_URL     = "https://login.microsoftonline.com/common/oauth2/token"
TS_EPD_URL       = "https://epdapi.tradesolution.no/v2/gtins/{gtin}"
TS_MEDIA_URL     = "https://mediastore.tradesolution.no/api/images/{gtin}"

_ts_token_cache = {}

def _get_ts_token():
    import urllib.parse
    if _ts_token_cache.get("token") and time.time() < _ts_token_cache.get("expires", 0):
        return _ts_token_cache["token"]
    body = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "client_id": TS_CLIENT_ID,
        "client_secret": TS_CLIENT_SECRET,
        "resource": "https://epdapi.tradesolution.no",
    }).encode()
    req = urllib.request.Request(TS_TOKEN_URL, data=body, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        tok = json.loads(r.read())
    _ts_token_cache["token"] = tok["access_token"]
    _ts_token_cache["expires"] = time.time() + int(tok.get("expires_in", 36000)) - 60
    return _ts_token_cache["token"]

def query_kassal(gtin, cache):
    """Query Kassal.app for Norwegian store availability, name, and image.

    Returns dict with:
      found        bool — product in any store
      names        list of unique names across stores
      stores       list of store names carrying the product
      image_url    best image URL (prefers ngdata.no)
      current_price float or None
      store_urls   {store_name: product_url}
    Returns {} if not configured or on error.
    """
    if not KASSAL_TOKEN:
        return {}
    cache_key = f"kassal_{gtin}"
    if cache_key in cache:
        return cache[cache_key]

    url = KASSAL_API.format(gtin=gtin)
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {KASSAL_TOKEN}", "Accept": "application/json",
                 "User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())
        products = data.get("data", {}).get("products") or []
        if not products:
            result = {"found": False, "names": [], "stores": [], "image_url": None,
                      "current_price": None, "store_urls": {}}
        else:
            names  = list(dict.fromkeys(p["name"] for p in products if p.get("name")))
            stores = list(dict.fromkeys(p["store"]["name"] for p in products if p.get("store")))
            # Prefer ngdata.no image (official store image)
            imgs = [p["image"] for p in products if p.get("image") and "ngdata.no" in (p.get("image") or "")]
            if not imgs:
                imgs = [p["image"] for p in products if p.get("image")]
            prices = [p["current_price"]["price"] for p in products
                      if p.get("current_price") and p["current_price"].get("price") is not None]
            urls = {p["store"]["name"]: p["url"] for p in products
                    if p.get("store") and p.get("url")}
            result = {
                "found": True,
                "names": names,
                "stores": stores,
                "image_url": imgs[0] if imgs else None,
                "current_price": min(prices) if prices else None,
                "store_urls": urls,
            }
    except urllib.error.HTTPError as e:
        result = {"found": False, "_error": f"HTTP {e.code}", "names": [], "stores": [],
                  "image_url": None, "current_price": None, "store_urls": {}}
    except Exception as e:
        result = {"found": False, "_error": str(e), "names": [], "stores": [],
                  "image_url": None, "current_price": None, "store_urls": {}}

    cache[cache_key] = result
    return result


def query_tradesolution(gtin, cache):
    """Returns {name, image_url} or {} if not configured / not found."""
    if not TS_CLIENT_ID or not TS_CLIENT_SECRET:
        return {}
    cache_key = f"ts_{gtin}"
    if cache_key in cache:
        return cache[cache_key]
    try:
        token = _get_ts_token()
        url = TS_EPD_URL.format(gtin=gtin)
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())
        name = data.get("productName") or data.get("name") or None
        img  = data.get("imageUrl") or None
        result = {"name": name, "image_url": img}
    except Exception as e:
        result = {"_error": str(e)}
    cache[cache_key] = result
    return result

# ---------------------------------------------------------------------------
# Perceptual hash (pHash, no extra deps)
# ---------------------------------------------------------------------------

def _open_gray32(path):
    try:
        from PIL import Image
        return Image.open(path).convert("L").resize((32, 32))
    except Exception:
        return None

def _fetch_image(url, cache_path):
    if cache_path.exists():
        return cache_path
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=12) as r:
            cache_path.write_bytes(r.read())
        return cache_path
    except Exception:
        return None

def phash(img):
    if img is None:
        return None
    try:
        px = list(img.getdata())
        mean = sum(px) / len(px)
        return int("".join("1" if p >= mean else "0" for p in px), 2)
    except Exception:
        return None

def hamming(h1, h2):
    if h1 is None or h2 is None:
        return None
    return bin(h1 ^ h2).count("1")

# ---------------------------------------------------------------------------
# Reference image quality
# ---------------------------------------------------------------------------

def blur_score(path):
    try:
        from PIL import Image, ImageFilter
        img = Image.open(path).convert("L")
        lap = img.filter(ImageFilter.FIND_EDGES)
        px = list(lap.getdata())
        mean = sum(px) / len(px)
        return round(sum((p - mean) ** 2 for p in px) / len(px), 1)
    except Exception:
        return None

def ref_image_path(code):
    d = REF_DIR / code
    for name in ("front.jpg", "main.jpg"):
        p = d / name
        if p.exists():
            return p
    return None

# ---------------------------------------------------------------------------
# Severity scoring
# ---------------------------------------------------------------------------

FLAG_SEVERITY = {
    "empty_product_name":           5,
    "invalid_ean":                  3,
    "nonstandard_ean":              2,
    "not_in_off":                   1,
    "name_mismatch_high":           3,   # OFF sim < 0.4
    "name_mismatch_medium":         2,   # OFF sim 0.4–0.65
    "kassal_name_mismatch_high":    4,   # Kassal sim < 0.4  (store data = more reliable)
    "kassal_name_mismatch_medium":  2,   # Kassal sim 0.4–0.65
    "discontinued":                 3,   # not in any store AND not in OFF
    "image_hash_mismatch":          3,   # hamming > 400
    "image_hash_suspect":           2,   # hamming 250–400
    "kassal_image_mismatch":        3,   # Kassal image hash > 400
    "kassal_image_suspect":         2,   # Kassal image hash 250–400
    "no_ref_image":                 2,
    "ref_image_blurry":             2,   # blur < 30
    "ref_image_suspect":            1,   # blur 30–60
    "no_product_code":              2,
}

FLAG_DESCRIPTION = {
    "empty_product_name":           "Product name is empty in dataset — EAN present but name missing",
    "invalid_ean":                  "EAN checksum failed — barcode may be corrupt or incorrectly entered",
    "nonstandard_ean":              "Non-standard code (PLU or truncated EAN)",
    "not_in_off":                   "GTIN not found in Open Food Facts",
    "name_mismatch_high":           "Our name vs OFF name similarity < 0.4 — possible wrong product code",
    "name_mismatch_medium":         "Our name vs OFF name similarity 0.4–0.65 — review recommended",
    "kassal_name_mismatch_high":    "Our name vs Kassal store name similarity < 0.4 — likely wrong label",
    "kassal_name_mismatch_medium":  "Our name vs Kassal store name similarity 0.4–0.65 — review recommended",
    "discontinued":                 "Product not found in any store (Kassal) AND not in OFF — likely discontinued",
    "image_hash_mismatch":          "Reference image vs OFF image hash distance > 400",
    "image_hash_suspect":           "Reference image vs OFF image hash distance 250–400",
    "kassal_image_mismatch":        "Reference image vs Kassal store image hash distance > 400",
    "kassal_image_suspect":         "Reference image vs Kassal store image hash distance 250–400",
    "no_ref_image":                 "No reference image found for this product",
    "ref_image_blurry":             "Reference image blur score < 30 — too low quality",
    "ref_image_suspect":            "Reference image blur score 30–60 — borderline quality",
    "no_product_code":              "Category has no product_code — cannot validate via EAN APIs",
}

def severity(flags):
    return sum(FLAG_SEVERITY.get(f, 1) for f in flags)

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

FINDINGS_COLS = [
    "run_date", "product_code", "product_name", "category_id",
    "flag_type", "flag_description", "severity",
    "gtin_type", "gtin_valid", "gtin_normalized",
    "off_found", "off_name", "off_name_similarity",
    "kassal_found", "kassal_name", "kassal_name_similarity",
    "kassal_stores", "kassal_price", "kassal_store_url",
    "discontinued",
    "phash_distance_off", "phash_distance_kassal", "blur_score",
    "annotation_count", "corrected_count",
    "annotation_ids_sample",   # first 20 annotation IDs (comma-sep)
    "image_ids_sample",        # first 20 unique image IDs (comma-sep)
    "notes",
]

ANNOTATIONS_COLS = [
    "run_date", "annotation_id", "image_id", "category_id",
    "product_code", "product_name",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "flag_types",              # pipe-separated flags for this category
    "severity",
    "notes",
]

def write_csv(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_HTML_HEAD = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Product Validation Audit</title>
<style>
body{font-family:monospace;background:#111;color:#eee;margin:0;padding:20px}
h1{color:#fff}.summary{background:#222;border-radius:6px;padding:16px;margin-bottom:24px}
.product{display:flex;gap:16px;border:1px solid #333;border-radius:6px;padding:16px;margin-bottom:16px;background:#1a1a1a}
.product.sev-critical{border-color:#8e0000}.product.sev-high{border-color:#c0392b}
.product.sev-medium{border-color:#e67e22}.product.sev-low{border-color:#2ecc71}.product.sev-ok{border-color:#27ae60;opacity:.6}
.images{display:flex;gap:8px;flex-wrap:wrap}
.images img{height:130px;object-fit:contain;background:#fff;border-radius:4px;border:2px solid #444}
.images img.bad{border-color:#c0392b}
.meta{flex:1;min-width:240px}.meta h3{margin:0 0 6px;font-size:13px}
.flags{margin:6px 0}
.flag{display:inline-block;padding:2px 6px;border-radius:3px;font-size:11px;margin:2px}
.flag.critical{background:#8e0000}.flag.high{background:#c0392b}.flag.medium{background:#e67e22}.flag.info{background:#2980b9}
table{border-collapse:collapse;width:100%;font-size:12px;margin-top:6px}
td,th{border:1px solid #444;padding:3px 7px;text-align:left}th{background:#222}
.ok{color:#2ecc71}.bad{color:#e74c3c}.warn{color:#f39c12}.na{color:#888}
.section{color:#aaa;margin:28px 0 10px;font-size:16px;border-bottom:1px solid #333;padding-bottom:6px}
</style></head><body><h1>Product Validation Audit</h1>
"""

def _flag_html(flag):
    lvl = "critical" if FLAG_SEVERITY.get(flag, 1) >= 5 else \
          "high"     if FLAG_SEVERITY.get(flag, 1) >= 3 else \
          "medium"   if FLAG_SEVERITY.get(flag, 1) >= 2 else "info"
    return f'<span class="flag {lvl}">{flag}</span>'

def _sev_cls(sev):
    if sev >= 5: return "sev-critical"
    if sev >= 3: return "sev-high"
    if sev >= 2: return "sev-medium"
    if sev >= 1: return "sev-low"
    return "sev-ok"

def _cv(val, good, bad, fmt=".0f"):
    if val is None: return '<span class="na">N/A</span>'
    cls = "ok" if val >= good else "bad" if val < bad else "warn"
    return f'<span class="{cls}">{val:{fmt}}</span>'

def _img(path, label, bad=False):
    if path is None or not Path(path).exists():
        return f'<div style="text-align:center;color:#888;font-size:10px;width:90px">{label}<br>[no image]</div>'
    cls = ' class="bad"' if bad else ""
    return f'<div style="text-align:center"><img src="{path}"{cls}><br><span style="font-size:10px;color:#aaa">{label}</span></div>'

def build_html(results, cats_no_code, run_date):
    parts = [_HTML_HEAD]

    # Summary table
    s_total   = len(results)
    s_flagged = sum(1 for r in results if r["flags"])
    counters  = {k: 0 for k in FLAG_SEVERITY}
    for r in results:
        for f in r["flags"]:
            if f in counters:
                counters[f] += 1

    rows = "".join(
        f'<tr><td>{k}</td><td class="{"bad" if v else "ok"}">{v}</td>'
        f'<td style="color:#888;font-size:11px">{FLAG_DESCRIPTION.get(k,"")}</td></tr>'
        for k, v in counters.items()
    )
    parts.append(f"""<div class="summary">
<b>Run date: {run_date}</b> &mdash; {s_total} products, {s_flagged} with flags,
{len(cats_no_code)} categories with no product_code
<table><tr><th>Flag</th><th>Count</th><th>Description</th></tr>{rows}
</table></div>""")

    # Categories with no product code
    parts.append(f'<div class="section">Categories with no product_code ({len(cats_no_code)})</div>')
    for c in sorted(cats_no_code, key=lambda x: -x["ann_count"]):
        parts.append(
            f'<div class="product sev-medium"><div class="meta"><h3>{c["category_name"]}</h3>'
            f'<span class="flag medium">no_product_code</span>'
            f'<table><tr><th>category_id</th><td>{c["category_id"]}</td>'
            f'<th>annotations</th><td>{c["ann_count"]}</td></tr></table></div></div>'
        )

    # Products grouped by severity
    for label, lo, hi in [("Critical (≥5)", 5, 999), ("High (3–4)", 3, 4),
                           ("Medium (2)", 2, 2), ("Low / OK (0–1)", 0, 1)]:
        group = [r for r in results if lo <= r["severity"] <= hi]
        if not group:
            continue
        parts.append(f'<div class="section">{label} — {len(group)} products</div>')
        for r in group:
            sev = r["severity"]
            flags_html = " ".join(_flag_html(f) for f in r["flags"]) or '<span class="flag info">clean</span>'
            our_img  = _img(r.get("ref_image_path"), "our reference",
                            bad="image_hash_mismatch" in r["flags"] or "kassal_image_mismatch" in r["flags"])
            off_img  = _img(r.get("off_image_path"), "OFF image",
                            bad="image_hash_mismatch" in r["flags"])
            ksl_img  = _img(r.get("kassal_image_path"), "store image",
                            bad="kassal_image_mismatch" in r["flags"])
            off_found = r["off_found"]
            ksl_found = r.get("kassal_found", False)
            ksl_stores = "|".join(r.get("kassal_stores") or []) or '<span class="na">not in any store</span>'
            ksl_price  = r.get("kassal_price")
            ksl_url    = r.get("kassal_store_url") or ""
            ksl_link   = f'<a href="{ksl_url}" target="_blank" style="color:#5dade2">{ksl_url[:70]}</a>' if ksl_url else '<span class="na">—</span>'
            disc_cls   = "bad" if r.get("discontinued") else "ok"
            h_ksl = r.get("phash_distance_kassal")
            h_off = r.get("phash_distance_off")

            parts.append(f"""<div class="product {_sev_cls(sev)}">
  <div class="meta">
    <h3>{r["product_name"] or "<i style='color:#e74c3c'>[EMPTY NAME]</i>"} &nbsp;
        <span style="color:#888;font-size:11px">sev={sev}</span></h3>
    <div class="flags">{flags_html}</div>
    <table>
      <tr><th>product_code</th><td>{r["product_code"]}</td>
          <th>type</th><td>{r["gtin_type"]}</td>
          <th>checksum</th><td class="{'ok' if r['gtin_valid'] else 'bad'}">{'✓' if r['gtin_valid'] else '✗'}</td>
          <th>category_id</th><td>{r.get("category_id","—")}</td></tr>
      <tr><th>our name</th><td colspan="4">{r["product_name"] or "<span class='bad'>⚠ EMPTY</span>"}</td></tr>
      <tr><th>OFF name</th><td colspan="2">{r["off_name"] or '<span class="na">—</span>'}</td>
          <th>OFF sim</th><td>{_cv(r.get("off_name_similarity"), 0.65, 0.4, ".3f")}</td>
          <th>found</th><td class="{'ok' if off_found else 'na'}">{'✓' if off_found else '✗'}</td></tr>
      <tr><th>Kassal name</th><td colspan="2">{r.get("kassal_name") or '<span class="na">—</span>'}</td>
          <th>Kassal sim</th><td>{_cv(r.get("kassal_name_similarity"), 0.65, 0.4, ".3f")}</td>
          <th>found</th><td class="{'ok' if ksl_found else 'na'}">{'✓' if ksl_found else '✗'}</td></tr>
      <tr><th>stores</th><td colspan="2">{ksl_stores}</td>
          <th>price</th><td>{f"{ksl_price} kr" if ksl_price else '<span class="na">—</span>'}</td>
          <th>discontinued</th><td class="{disc_cls}">{'⚠ YES' if r.get("discontinued") else 'no'}</td></tr>
      <tr><th>store URL</th><td colspan="5" style="font-size:11px">{ksl_link}</td></tr>
      <tr><th>hash Kassal</th><td>{_cv(h_ksl, 0, 400) if h_ksl is not None else '<span class="na">N/A</span>'}</td>
          <th>hash OFF</th><td>{_cv(h_off, 0, 400) if h_off is not None else '<span class="na">N/A</span>'}</td>
          <th>blur</th><td>{_cv(r["blur_score"], 60, 30)}</td>
          <th>annotations</th><td>{r["annotation_count"]} ({r["corrected_count"]} corr.)</td></tr>
    </table>
  </div>
  <div class="images">{our_img}{ksl_img}{off_img}</div>
</div>""")

    parts.append("</body></html>")
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-ok", action="store_true")
    args = parser.parse_args()

    from datetime import date
    run_date = date.today().isoformat()

    print("Loading data...")
    with open(META_PATH) as f:
        meta = json.load(f)
    with open(ANN_PATH) as f:
        ann_data = json.load(f)

    products    = meta["products"]
    categories  = ann_data["categories"]
    annotations = ann_data["annotations"]

    # Indexes
    cat_name_to_id = {c["name"]: c["id"] for c in categories}
    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    # category_id -> [annotation]
    anns_by_cat = defaultdict(list)
    for a in annotations:
        anns_by_cat[a["category_id"]].append(a)

    if args.limit:
        products = products[:args.limit]

    off_cache = load_off_cache()
    results   = []

    print(f"Validating {len(products)} products...")
    for i, p in enumerate(products):
        code     = p["product_code"]
        our_name = p["product_name"].strip()
        ann_cnt  = p.get("annotation_count", 0)
        corr_cnt = p.get("corrected_count", 0)

        flags = []

        # 0. Empty name
        if not our_name:
            flags.append("empty_product_name")

        # 1. EAN
        ginfo = validate_gtin(code)
        if not ginfo["valid"]:
            flags.append("invalid_ean" if ginfo["type"] in ("EAN-13","UPC-A","EAN-8") else "nonstandard_ean")

        # 2. OFF
        queryable = ginfo["type"] in ("EAN-13","UPC-A","EAN-8","EAN-11(truncated)")
        query_code = ginfo["normalized"] or code
        # Kassal uses the raw/original code — it stores 11-digit EANs as-is,
        # so zero-padding to 13 digits causes 404s on truncated codes.
        kassal_code = code
        if queryable:
            print(f"  [{i+1}/{len(products)}] OFF+Kassal: {code}  {our_name[:40]}")
            off = query_off(query_code, off_cache)
            time.sleep(OFF_RATE_S)
        else:
            off = {}
            print(f"  [{i+1}/{len(products)}] Kassal only (non-std EAN): {code}  {our_name[:40]}")

        off_found = bool(off and "_error" not in off)
        off_n     = off_name(off) if off_found else None
        off_img_u = off_img_url(off) if off_found else None

        if queryable and not off_found:
            flags.append("not_in_off")

        # 3. OFF name similarity
        off_sim = None
        if off_n and our_name:
            off_sim = name_similarity(our_name, off_n)
            if off_sim < 0.4:
                flags.append("name_mismatch_high")
            elif off_sim < 0.65:
                flags.append("name_mismatch_medium")

        # 4. Kassal (Norwegian store check) — use original code, not zero-padded
        ksl = query_kassal(kassal_code, off_cache)
        time.sleep(KASSAL_RATE_S)
        ksl_found  = ksl.get("found", False)
        ksl_names  = ksl.get("names", [])
        ksl_stores = ksl.get("stores", [])
        ksl_img_u  = ksl.get("image_url")
        ksl_price  = ksl.get("current_price")
        ksl_urls   = ksl.get("store_urls", {})
        # Best Kassal name = first (most common across stores)
        ksl_name = ksl_names[0] if ksl_names else None

        ksl_sim = None
        if ksl_name and our_name:
            ksl_sim = name_similarity(our_name, ksl_name)
            if ksl_sim < 0.4:
                flags.append("kassal_name_mismatch_high")
            elif ksl_sim < 0.65:
                flags.append("kassal_name_mismatch_medium")

        # Discontinued = not in Kassal AND not in OFF
        is_discontinued = queryable and not ksl_found and not off_found
        if is_discontinued:
            flags.append("discontinued")

        # 5. Tradesolution (no-op until credentials set)
        ts = query_tradesolution(query_code, off_cache)
        ts_name = ts.get("name")

        # 6. Reference image quality
        ref_path = ref_image_path(code)
        if ref_path is None:
            flags.append("no_ref_image")
            bs = None
        else:
            bs = blur_score(ref_path)
            if bs is not None:
                if bs < 30:
                    flags.append("ref_image_blurry")
                elif bs < 60:
                    flags.append("ref_image_suspect")

        # 7. Image hash vs OFF
        phash_dist_off = None
        off_img_local  = None
        if off_img_u and ref_path:
            cache_img = IMG_CACHE_DIR / f"{query_code}.jpg"
            local = _fetch_image(off_img_u, cache_img)
            off_img_local = str(cache_img) if local else None
            h_ref = phash(_open_gray32(ref_path))
            h_off = phash(_open_gray32(cache_img)) if local else None
            phash_dist_off = hamming(h_ref, h_off)
            if phash_dist_off is not None:
                if phash_dist_off > 400:
                    flags.append("image_hash_mismatch")
                elif phash_dist_off > 250:
                    flags.append("image_hash_suspect")

        # 8. Image hash vs Kassal (more reliable — uses actual store images)
        phash_dist_ksl = None
        ksl_img_local  = None
        if ksl_img_u and ref_path:
            ksl_cache_img = KSL_IMG_DIR / f"{kassal_code}.jpg"
            ksl_local = _fetch_image(ksl_img_u, ksl_cache_img)
            ksl_img_local = str(ksl_cache_img) if ksl_local else None
            h_ref = phash(_open_gray32(ref_path))
            h_ksl = phash(_open_gray32(ksl_cache_img)) if ksl_local else None
            phash_dist_ksl = hamming(h_ref, h_ksl)
            if phash_dist_ksl is not None:
                if phash_dist_ksl > 400:
                    flags.append("kassal_image_mismatch")
                elif phash_dist_ksl > 250:
                    flags.append("kassal_image_suspect")

        # 9. Category linkage
        cat_id = cat_name_to_id.get(our_name)
        cat_anns = anns_by_cat.get(cat_id, []) if cat_id is not None else []

        # Pick best store URL for CSV (prefer MENY)
        best_url = ksl_urls.get("Meny") or ksl_urls.get("KIWI") or ksl_urls.get("SPAR") or \
                   (list(ksl_urls.values())[0] if ksl_urls else "")

        results.append({
            "run_date":             run_date,
            "product_code":         code,
            "product_name":         our_name,
            "category_id":          cat_id,
            "flags":                flags,
            "severity":             severity(flags),
            "gtin_type":            ginfo["type"],
            "gtin_valid":           ginfo["valid"],
            "gtin_normalized":      query_code,
            "off_found":            off_found,
            "off_name":             off_n,
            "off_name_similarity":  off_sim,
            "kassal_found":         ksl_found,
            "kassal_name":          ksl_name,
            "kassal_names":         ksl_names,
            "kassal_name_similarity": ksl_sim,
            "kassal_stores":        ksl_stores,
            "kassal_price":         ksl_price,
            "kassal_store_url":     best_url,
            "discontinued":         is_discontinued,
            "ts_name":              ts_name,
            "phash_distance_off":   phash_dist_off,
            "phash_distance_kassal":phash_dist_ksl,
            "blur_score":           bs,
            "annotation_count":     ann_cnt,
            "corrected_count":      corr_cnt,
            "ref_image_path":       str(ref_path) if ref_path else None,
            "off_image_path":       off_img_local,
            "kassal_image_path":    ksl_img_local,
            "cat_anns":             cat_anns,
        })

    save_off_cache(off_cache)

    # Categories with no product code
    product_names = {p["product_name"].strip() for p in meta["products"]}
    cats_no_code = [
        {
            "category_id":   c["id"],
            "category_name": c["name"],
            "ann_count":     len(anns_by_cat[c["id"]]),
        }
        for c in categories if c["name"] not in product_names
    ]

    results.sort(key=lambda r: -r["severity"])

    # ------------------------------------------------------------------
    # CSV 1: findings.csv — one row per (product × flag)
    # ------------------------------------------------------------------
    finding_rows = []
    for r in results:
        if not r["flags"]:
            # Still write one "clean" row per product so the CSV is complete
            cat_anns = r["cat_anns"]
            ann_ids = ",".join(str(a["id"]) for a in cat_anns[:20])
            img_ids = ",".join(str(i) for i in sorted({a["image_id"] for a in cat_anns})[:20])
            finding_rows.append({
                "run_date":              r["run_date"],
                "product_code":          r["product_code"],
                "product_name":          r["product_name"],
                "category_id":           r["category_id"],
                "flag_type":             "CLEAN",
                "flag_description":      "No issues found",
                "severity":              0,
                "gtin_type":             r["gtin_type"],
                "gtin_valid":            r["gtin_valid"],
                "gtin_normalized":       r["gtin_normalized"],
                "off_found":             r["off_found"],
                "off_name":              r["off_name"],
                "off_name_similarity":   r["off_name_similarity"],
                "kassal_found":          r["kassal_found"],
                "kassal_name":           r["kassal_name"],
                "kassal_name_similarity":r["kassal_name_similarity"],
                "kassal_stores":         "|".join(r["kassal_stores"]),
                "kassal_price":          r["kassal_price"],
                "kassal_store_url":      r["kassal_store_url"],
                "discontinued":          r["discontinued"],
                "phash_distance_off":    r["phash_distance_off"],
                "phash_distance_kassal": r["phash_distance_kassal"],
                "blur_score":            r["blur_score"],
                "annotation_count":      r["annotation_count"],
                "corrected_count":       r["corrected_count"],
                "annotation_ids_sample": ann_ids,
                "image_ids_sample":      img_ids,
                "notes":                 "",
            })
        else:
            for flag in r["flags"]:
                cat_anns = r["cat_anns"]
                ann_ids = ",".join(str(a["id"]) for a in cat_anns[:20])
                img_ids = ",".join(str(i) for i in sorted({a["image_id"] for a in cat_anns})[:20])
                finding_rows.append({
                    "run_date":              r["run_date"],
                    "product_code":          r["product_code"],
                    "product_name":          r["product_name"],
                    "category_id":           r["category_id"],
                    "flag_type":             flag,
                    "flag_description":      FLAG_DESCRIPTION.get(flag, ""),
                    "severity":              FLAG_SEVERITY.get(flag, 1),
                    "gtin_type":             r["gtin_type"],
                    "gtin_valid":            r["gtin_valid"],
                    "gtin_normalized":       r["gtin_normalized"],
                    "off_found":             r["off_found"],
                    "off_name":              r["off_name"],
                    "off_name_similarity":   r["off_name_similarity"],
                    "kassal_found":          r["kassal_found"],
                    "kassal_name":           r["kassal_name"],
                    "kassal_name_similarity":r["kassal_name_similarity"],
                    "kassal_stores":         "|".join(r["kassal_stores"]),
                    "kassal_price":          r["kassal_price"],
                    "kassal_store_url":      r["kassal_store_url"],
                    "discontinued":          r["discontinued"],
                    "phash_distance_off":    r["phash_distance_off"],
                    "phash_distance_kassal": r["phash_distance_kassal"],
                    "blur_score":            r["blur_score"],
                    "annotation_count":      r["annotation_count"],
                    "corrected_count":       r["corrected_count"],
                    "annotation_ids_sample": ann_ids,
                    "image_ids_sample":      img_ids,
                    "notes":                 "",
                })

    # Rows for categories with no product code
    for c in cats_no_code:
        cat_anns = anns_by_cat[c["category_id"]]
        ann_ids = ",".join(str(a["id"]) for a in cat_anns[:20])
        img_ids = ",".join(str(i) for i in sorted({a["image_id"] for a in cat_anns})[:20])
        finding_rows.append({
            "run_date":              run_date,
            "product_code":          "",
            "product_name":          c["category_name"],
            "category_id":           c["category_id"],
            "flag_type":             "no_product_code",
            "flag_description":      FLAG_DESCRIPTION["no_product_code"],
            "severity":              FLAG_SEVERITY["no_product_code"],
            "gtin_type":             "",
            "gtin_valid":            "",
            "gtin_normalized":       "",
            "off_found":             False,
            "off_name":              "",
            "off_name_similarity":   "",
            "kassal_found":          False,
            "kassal_name":           "",
            "kassal_name_similarity":"",
            "kassal_stores":         "",
            "kassal_price":          "",
            "kassal_store_url":      "",
            "discontinued":          "",
            "phash_distance_off":    "",
            "phash_distance_kassal": "",
            "blur_score":            "",
            "annotation_count":      c["ann_count"],
            "corrected_count":       "",
            "annotation_ids_sample": ann_ids,
            "image_ids_sample":      img_ids,
            "notes":                 "",
        })

    findings_path = OUT_DIR / "findings.csv"
    write_csv(findings_path, FINDINGS_COLS, finding_rows)
    print(f"findings.csv: {len(finding_rows)} rows -> {findings_path}")

    # ------------------------------------------------------------------
    # CSV 2: annotations_flagged.csv — one row per annotation in flagged categories
    # ------------------------------------------------------------------
    ann_rows = []
    flagged_results = [r for r in results if r["flags"]]
    # Also include cats_no_code
    flagged_cat_ids = {r["category_id"] for r in flagged_results if r["category_id"] is not None}
    flagged_cat_ids |= {c["category_id"] for c in cats_no_code}

    for r in flagged_results:
        if r["category_id"] is None:
            continue
        flag_str = "|".join(r["flags"])
        for a in r["cat_anns"]:
            bbox = a.get("bbox", [None, None, None, None])
            ann_rows.append({
                "run_date":      run_date,
                "annotation_id": a["id"],
                "image_id":      a["image_id"],
                "category_id":   a["category_id"],
                "product_code":  r["product_code"],
                "product_name":  r["product_name"],
                "bbox_x":        bbox[0],
                "bbox_y":        bbox[1],
                "bbox_w":        bbox[2],
                "bbox_h":        bbox[3],
                "flag_types":    flag_str,
                "severity":      r["severity"],
                "notes":         "",
            })

    for c in cats_no_code:
        for a in anns_by_cat[c["category_id"]]:
            bbox = a.get("bbox", [None, None, None, None])
            ann_rows.append({
                "run_date":      run_date,
                "annotation_id": a["id"],
                "image_id":      a["image_id"],
                "category_id":   a["category_id"],
                "product_code":  "",
                "product_name":  c["category_name"],
                "bbox_x":        bbox[0],
                "bbox_y":        bbox[1],
                "bbox_w":        bbox[2],
                "bbox_h":        bbox[3],
                "flag_types":    "no_product_code",
                "severity":      FLAG_SEVERITY["no_product_code"],
                "notes":         "",
            })

    ann_rows.sort(key=lambda r: (-r["severity"], r["category_id"] or 0))
    anns_path = OUT_DIR / "annotations_flagged.csv"
    write_csv(anns_path, ANNOTATIONS_COLS, ann_rows)
    print(f"annotations_flagged.csv: {len(ann_rows)} rows -> {anns_path}")

    # ------------------------------------------------------------------
    # JSON report
    # ------------------------------------------------------------------
    report = {
        "run_date":   run_date,
        "summary": {
            "total_products":             len(results),
            "with_flags":                 sum(1 for r in results if r["flags"]),
            "categories_no_product_code": len(cats_no_code),
            "flagged_annotations":        len(ann_rows),
            **{k: sum(1 for r in results if k in r["flags"]) for k in FLAG_SEVERITY},
        },
        "products":                    [{k: v for k, v in r.items() if k != "cat_anns"} for r in results],
        "categories_without_product_code": cats_no_code,
    }
    json_path = OUT_DIR / "report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report.json -> {json_path}")

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------
    html_path = OUT_DIR / "report.html"
    html_path.write_text(build_html(results, cats_no_code, run_date))
    print(f"report.html -> {html_path}")

    # ------------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------------
    s = report["summary"]
    print(f"\n{'='*50}")
    print(f"Run date       : {run_date}")
    print(f"Products       : {s['total_products']}")
    print(f"With flags     : {s['with_flags']}")
    print(f"Flagged anns   : {s['flagged_annotations']}")
    print(f"{'='*50}")
    for k in FLAG_SEVERITY:
        v = s.get(k, 0)
        if v:
            print(f"  {k:<30} {v}")
    print(f"  {'no_product_code':<30} {s['categories_no_product_code']}")


if __name__ == "__main__":
    main()
