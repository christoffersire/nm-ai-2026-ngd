"""
Generate an HTML page for manual review of suspicious annotations.

Crops each annotation, embeds images as base64, and creates an interactive
page where you can mark each annotation as CORRECT or WRONG.

Usage:
  python data/scripts/generate_review_html.py
  python data/scripts/generate_review_html.py --max 500 --min-suspicion 0.05
"""
import argparse
import base64
import io
import json
from pathlib import Path
from collections import defaultdict

from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REF_DIR = RAW_DIR / "product_images"


def img_to_base64(img, max_size=200, quality=80):
    """Convert PIL Image to base64 JPEG string."""
    img = img.convert("RGB")
    w, h = img.size
    if w > max_size or h > max_size:
        ratio = min(max_size / w, max_size / h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def get_ref_b64(cat_id, cat_to_code, max_size=180, api_ean_map=None):
    """Get base64 reference image for a category.
    Checks metadata EAN first, then API-discovered EAN."""
    codes_to_try = []
    if cat_id in cat_to_code:
        codes_to_try.append(cat_to_code[cat_id])
    if api_ean_map and cat_id in api_ean_map:
        codes_to_try.append(api_ean_map[cat_id])

    for code in codes_to_try:
        for name in ["main.jpg", "front.jpg", "back.jpg"]:
            p = REF_DIR / code / name
            if p.exists():
                try:
                    return img_to_base64(Image.open(p), max_size=max_size)
                except Exception:
                    continue
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None,
                        help="Maximum number of items to include")
    parser.add_argument("--min-suspicion", type=float, default=0.0,
                        help="Minimum suspicion score to include")
    parser.add_argument("--output", default=str(DATA_DIR / "review.html"))
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    with open(RAW_DIR / "annotations.json") as f:
        data = json.load(f)
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    with open(DATA_DIR / "all_similarities.json") as f:
        all_sims = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    cat_name_to_id = {c["name"]: c["id"] for c in data["categories"]}
    id_to_img = {img["id"]: img for img in data["images"]}
    ann_by_id = {a["id"]: a for a in data["annotations"]}

    cat_to_code = {}
    corrected_by_cat = {}
    for p in meta["products"]:
        if p["product_name"] in cat_name_to_id:
            cid = cat_name_to_id[p["product_name"]]
            cat_to_code[cid] = p["product_code"]
            corrected_by_cat[cid] = p.get("corrected_count", 0)

    # Build API EAN map from verification data (for categories not in metadata)
    api_ean_map = {}
    verif_path = DATA_DIR / "product_verification.json"
    if verif_path.exists():
        with open(verif_path) as f:
            verif_data = json.load(f)
        for r in verif_data.get("results", []):
            if r.get("api_ean") and r["cat_id"] not in cat_to_code:
                api_ean_map[r["cat_id"]] = r["api_ean"]

    # Filter to suspicious items
    # Separate: items WITH reference (can compare) vs WITHOUT (no reference image)
    has_ref = [s for s in all_sims if s["gt_sim"] >= 0 and s["suspicion"] >= args.min_suspicion]
    no_ref = [s for s in all_sims if s["gt_sim"] < 0]

    # Sort has_ref by suspicion (most suspicious first)
    # Sort no_ref by suspicion too, but they go AFTER has_ref items
    has_ref.sort(key=lambda x: -x["suspicion"])
    no_ref.sort(key=lambda x: -x["suspicion"])
    candidates = has_ref + no_ref

    if args.max:
        candidates = candidates[:args.max]

    print(f"Generating HTML for {len(candidates)} candidates (suspicion >= {args.min_suspicion})...")

    # Pre-compute reference images (cache by cat_id)
    ref_cache = {}
    needed_cats = set()
    for c in candidates:
        needed_cats.add(c["gt_cat"])
        needed_cats.add(c["top1_cat"])
        needed_cats.add(c["top2_cat"])
        needed_cats.add(c["top3_cat"])

    print(f"Loading reference images for {len(needed_cats)} categories...")
    for cat_id in needed_cats:
        if cat_id not in ref_cache:
            ref_cache[cat_id] = get_ref_b64(cat_id, cat_to_code, api_ean_map=api_ean_map)

    # Build items with embedded images
    print("Cropping annotations and encoding images...")
    items = []
    img_cache = {}  # image_id -> PIL Image

    for i, cand in enumerate(candidates):
        ann_id = cand["ann_id"]
        ann = ann_by_id[ann_id]
        img_id = ann["image_id"]

        if img_id not in img_cache:
            img_info = id_to_img[img_id]
            img_path = RAW_DIR / "images" / img_info["file_name"]
            img_cache[img_id] = Image.open(img_path).convert("RGB")

        img = img_cache[img_id]
        x, y, w, h = ann["bbox"]
        img_w, img_h = img.size
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(img_w, int(x + w)), min(img_h, int(y + h))

        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        crop = img.crop((x1, y1, x2, y2))
        crop_b64 = img_to_base64(crop, max_size=250, quality=85)

        gt_name = cat_id_to_name.get(cand["gt_cat"], f"Unknown ({cand['gt_cat']})")
        top1_name = cat_id_to_name.get(cand["top1_cat"], f"Unknown ({cand['top1_cat']})")
        top2_name = cat_id_to_name.get(cand["top2_cat"], f"Unknown ({cand['top2_cat']})")
        top3_name = cat_id_to_name.get(cand["top3_cat"], f"Unknown ({cand['top3_cat']})")
        img_name = id_to_img[img_id]["file_name"]
        corr_count = corrected_by_cat.get(cand["gt_cat"], "N/A")

        items.append({
            "ann_id": ann_id,
            "img_name": img_name,
            "gt_cat": cand["gt_cat"],
            "gt_name": gt_name,
            "gt_sim": cand["gt_sim"],
            "corr_count": corr_count,
            "suspicion": round(cand["suspicion"], 4),
            "crop_b64": crop_b64,
            "gt_ref_b64": ref_cache.get(cand["gt_cat"]),
            "top1_cat": cand["top1_cat"],
            "top1_name": top1_name,
            "top1_sim": cand["top1_sim"],
            "top1_ref_b64": ref_cache.get(cand["top1_cat"]),
            "top2_cat": cand["top2_cat"],
            "top2_name": top2_name,
            "top2_sim": cand["top2_sim"],
            "top2_ref_b64": ref_cache.get(cand["top2_cat"]),
            "top3_cat": cand["top3_cat"],
            "top3_name": top3_name,
            "top3_sim": cand["top3_sim"],
            "top3_ref_b64": ref_cache.get(cand["top3_cat"]),
            "bbox": ann["bbox"],
        })

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(candidates)}...")
            # Free memory for images we won't need again
            if len(img_cache) > 20:
                img_cache.clear()

    print(f"Generated {len(items)} review items")

    # Generate HTML
    html = _generate_html(items)

    with open(args.output, "w") as f:
        f.write(html)
    print(f"HTML saved to: {args.output}")
    print(f"File size: {Path(args.output).stat().st_size / 1024 / 1024:.1f} MB")


def _generate_html(items):
    items_json = json.dumps(items)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NorgesGruppen Annotation Review</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f0f2f5; color: #333; }}

.header {{
    background: #1a1a2e; color: white; padding: 16px 24px;
    display: flex; justify-content: space-between; align-items: center;
    position: sticky; top: 0; z-index: 100;
}}
.header h1 {{ font-size: 18px; }}
.progress {{ font-size: 14px; opacity: 0.9; }}
.progress-bar {{
    width: 200px; height: 8px; background: #333; border-radius: 4px;
    display: inline-block; vertical-align: middle; margin-left: 8px;
}}
.progress-fill {{ height: 100%; background: #4CAF50; border-radius: 4px; transition: width 0.3s; }}

.controls {{
    display: flex; gap: 8px; align-items: center;
}}
.controls button {{
    padding: 6px 14px; border: none; border-radius: 4px;
    cursor: pointer; font-size: 13px; font-weight: 500;
}}
.btn-export {{ background: #4CAF50; color: white; }}
.btn-export:hover {{ background: #45a049; }}
.btn-undo {{ background: #ff9800; color: white; }}
.btn-undo:hover {{ background: #f57c00; }}
.btn-skip {{ background: #9e9e9e; color: white; }}

.card {{
    max-width: 900px; margin: 20px auto; background: white;
    border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    overflow: hidden;
}}

.card-header {{
    padding: 16px 24px; background: #f8f9fa; border-bottom: 1px solid #e0e0e0;
    display: flex; justify-content: space-between; align-items: center;
}}
.card-number {{ font-size: 13px; color: #666; }}
.suspicion-badge {{
    padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600;
}}
.suspicion-high {{ background: #ffebee; color: #c62828; }}
.suspicion-med {{ background: #fff3e0; color: #e65100; }}
.suspicion-low {{ background: #e8f5e9; color: #2e7d32; }}

.question-box {{
    padding: 20px 24px; background: #e3f2fd; border-bottom: 1px solid #bbdefb;
}}
.question {{
    font-size: 20px; font-weight: 600; color: #1565c0; text-align: center;
    line-height: 1.4;
}}
.question .product-name {{
    display: inline-block; background: #1565c0; color: white;
    padding: 2px 12px; border-radius: 6px; margin: 4px 0;
}}

.comparison {{
    display: flex; padding: 24px; gap: 24px; justify-content: center;
    align-items: flex-start; flex-wrap: wrap;
}}

.image-panel {{
    text-align: center; flex-shrink: 0;
}}
.image-panel img {{
    border-radius: 8px; border: 3px solid #e0e0e0;
    max-width: 220px; max-height: 220px; display: block; margin: 0 auto;
}}
.image-panel .label {{
    margin-top: 8px; font-size: 13px; font-weight: 600; color: #666;
}}
.image-panel .sublabel {{
    font-size: 11px; color: #999; margin-top: 2px;
}}

.crop-panel img {{ border-color: #2196F3; }}
.ref-panel img {{ border-color: #FF9800; }}

.arrow {{
    font-size: 40px; color: #ccc; align-self: center;
    padding: 0 8px;
}}

.actions {{
    padding: 20px 24px; display: flex; gap: 12px;
    justify-content: center; border-top: 1px solid #e0e0e0;
    background: #fafafa;
}}

.action-btn {{
    padding: 14px 32px; border: none; border-radius: 8px;
    font-size: 16px; font-weight: 700; cursor: pointer;
    transition: transform 0.1s, box-shadow 0.1s;
    min-width: 200px;
}}
.action-btn:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
.action-btn:active {{ transform: translateY(0); }}

.btn-correct {{
    background: #4CAF50; color: white;
}}
.btn-wrong {{
    background: #f44336; color: white;
}}

.keyboard-hint {{
    text-align: center; padding: 8px; font-size: 12px; color: #999;
}}

/* Alternatives panel (shown when WRONG is clicked) */
.alternatives {{
    padding: 20px 24px; border-top: 2px solid #f44336;
    background: #fff8f8; display: none;
}}
.alternatives h3 {{
    font-size: 16px; margin-bottom: 16px; color: #c62828;
}}
.alt-grid {{
    display: flex; gap: 16px; justify-content: center; flex-wrap: wrap;
}}
.alt-option {{
    text-align: center; cursor: pointer; padding: 12px;
    border: 2px solid #e0e0e0; border-radius: 8px;
    transition: border-color 0.2s, background 0.2s;
    max-width: 180px;
}}
.alt-option:hover {{ border-color: #4CAF50; background: #f1f8e9; }}
.alt-option img {{
    max-width: 140px; max-height: 140px; border-radius: 4px;
    display: block; margin: 0 auto;
}}
.alt-option .alt-name {{ font-size: 12px; margin-top: 6px; font-weight: 500; }}
.alt-option .alt-sim {{ font-size: 11px; color: #999; }}
.alt-option .alt-key {{ font-size: 10px; color: #2196F3; font-weight: 700; }}

.alt-skip {{
    margin-top: 12px; text-align: center;
}}
.alt-skip button {{
    padding: 8px 24px; background: #9e9e9e; color: white;
    border: none; border-radius: 6px; cursor: pointer; font-size: 13px;
}}

.done-screen {{
    text-align: center; padding: 60px 24px;
    max-width: 600px; margin: 40px auto;
    background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}}
.done-screen h2 {{ color: #4CAF50; margin-bottom: 16px; }}

.hidden {{ display: none !important; }}
</style>
</head>
<body>

<div class="header">
    <h1>Annotation Review — NorgesGruppen Data</h1>
    <div class="progress">
        <span id="progress-text">0 / 0</span>
        <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
        <span id="stats-text" style="margin-left: 12px;"></span>
    </div>
    <div class="controls">
        <button class="btn-undo" onclick="undo()">↩ Undo (Z)</button>
        <button class="btn-export" onclick="exportResults()">💾 Export Results</button>
    </div>
</div>

<div id="card-container"></div>

<div id="done-screen" class="done-screen hidden">
    <h2>Review Complete!</h2>
    <p id="done-stats"></p>
    <br>
    <button class="btn-export" onclick="exportResults()" style="font-size: 18px; padding: 14px 32px;">
        💾 Export Corrections
    </button>
</div>

<script>
const ALL_ITEMS = {items_json};

let currentIndex = 0;
let decisions = {{}};  // ann_id -> {{ decision: "correct"|"wrong", new_cat: int|null }}
let history = [];  // stack of ann_ids for undo

// Load saved progress
try {{
    const saved = localStorage.getItem("annotation_review_progress");
    if (saved) {{
        const parsed = JSON.parse(saved);
        decisions = parsed.decisions || {{}};
        currentIndex = parsed.currentIndex || 0;
        history = parsed.history || [];
    }}
}} catch(e) {{}}

function saveProgress() {{
    localStorage.setItem("annotation_review_progress", JSON.stringify({{
        decisions, currentIndex, history
    }}));
}}

function updateProgress() {{
    const total = ALL_ITEMS.length;
    const reviewed = Object.keys(decisions).length;
    const correct = Object.values(decisions).filter(d => d.decision === "correct").length;
    const wrong = Object.values(decisions).filter(d => d.decision === "wrong").length;

    document.getElementById("progress-text").textContent = `${{reviewed}} / ${{total}}`;
    document.getElementById("progress-fill").style.width = `${{(reviewed / total) * 100}}%`;
    document.getElementById("stats-text").textContent = `✓ ${{correct}} correct, ✗ ${{wrong}} wrong`;
}}

function renderCard() {{
    // Skip already-reviewed items
    while (currentIndex < ALL_ITEMS.length && decisions[ALL_ITEMS[currentIndex].ann_id]) {{
        currentIndex++;
    }}

    if (currentIndex >= ALL_ITEMS.length) {{
        document.getElementById("card-container").innerHTML = "";
        document.getElementById("done-screen").classList.remove("hidden");
        const correct = Object.values(decisions).filter(d => d.decision === "correct").length;
        const wrong = Object.values(decisions).filter(d => d.decision === "wrong").length;
        document.getElementById("done-stats").textContent =
            `Reviewed ${{Object.keys(decisions).length}} annotations: ${{correct}} correct, ${{wrong}} wrong.`;
        return;
    }}

    const item = ALL_ITEMS[currentIndex];
    const suspicionClass = item.suspicion > 0.2 ? "suspicion-high" :
                           item.suspicion > 0.1 ? "suspicion-med" : "suspicion-low";

    const cropImg = item.crop_b64 ? `<img src="data:image/jpeg;base64,${{item.crop_b64}}" alt="Annotation crop">` : '<div style="width:200px;height:200px;background:#eee;display:flex;align-items:center;justify-content:center;">No image</div>';
    const refImg = item.gt_ref_b64 ? `<img src="data:image/jpeg;base64,${{item.gt_ref_b64}}" alt="Reference product">` : '<div style="width:180px;height:180px;background:#fff3e0;display:flex;align-items:center;justify-content:center;border-radius:8px;border:3px solid #FF9800;font-size:13px;color:#e65100;padding:12px;text-align:center;">No reference image available for this product</div>';

    const alt1Img = item.top1_ref_b64 ? `<img src="data:image/jpeg;base64,${{item.top1_ref_b64}}">` : '<div style="width:140px;height:140px;background:#eee;display:flex;align-items:center;justify-content:center;font-size:11px;">No image</div>';
    const alt2Img = item.top2_ref_b64 ? `<img src="data:image/jpeg;base64,${{item.top2_ref_b64}}">` : '<div style="width:140px;height:140px;background:#eee;display:flex;align-items:center;justify-content:center;font-size:11px;">No image</div>';
    const alt3Img = item.top3_ref_b64 ? `<img src="data:image/jpeg;base64,${{item.top3_ref_b64}}">` : '<div style="width:140px;height:140px;background:#eee;display:flex;align-items:center;justify-content:center;font-size:11px;">No image</div>';

    document.getElementById("card-container").innerHTML = `
        <div class="card">
            <div class="card-header">
                <span class="card-number">Item ${{currentIndex + 1}} of ${{ALL_ITEMS.length}} &mdash; ann_id: ${{item.ann_id}} &mdash; ${{item.img_name}}</span>
                <span class="suspicion-badge ${{suspicionClass}}">suspicion: ${{item.suspicion.toFixed(3)}}</span>
            </div>

            <div class="question-box">
                <div class="question">
                    Look at the <span style="color:#2196F3;">crop on the left</span>.<br>
                    It is currently labeled as:<br>
                    <span class="product-name">${{item.gt_name}}</span><br>
                    The <span style="color:#FF9800;">reference image on the right</span> shows what that product looks like.<br><br>
                    <strong>Is this label correct? Does the crop show this product?</strong>
                </div>
            </div>

            <div class="comparison">
                <div class="image-panel crop-panel">
                    ${{cropImg}}
                    <div class="label" style="color:#2196F3;">⬆ This is the crop from the training image</div>
                    <div class="sublabel">This is what the model will learn from</div>
                </div>
                <div class="arrow">⟷</div>
                <div class="image-panel ref-panel">
                    ${{refImg}}
                    <div class="label" style="color:#FF9800;">⬆ This is the reference for "${{item.gt_name}}"</div>
                    <div class="sublabel">cat_id: ${{item.gt_cat}} | similarity: ${{item.gt_sim.toFixed(3)}}</div>
                </div>
            </div>

            <div class="actions">
                <button class="action-btn btn-correct" onclick="markCorrect()">
                    ✓ YES, this is correct<br>
                    <small style="font-weight:normal;font-size:12px;">The crop matches the product (press Y)</small>
                </button>
                <button class="action-btn btn-wrong" onclick="showAlternatives()">
                    ✗ NO, this is wrong<br>
                    <small style="font-weight:normal;font-size:12px;">The crop does NOT match (press N)</small>
                </button>
            </div>

            <div class="keyboard-hint">Keyboard: <b>Y</b> = correct, <b>N</b> = wrong, <b>Z</b> = undo, <b>S</b> = skip</div>

            <div class="alternatives" id="alternatives-panel">
                <h3>What product does this crop ACTUALLY show? Pick the best match:</h3>
                <div class="alt-grid">
                    <div class="alt-option" onclick="markWrong(${{item.top1_cat}})">
                        ${{alt1Img}}
                        <div class="alt-name">${{item.top1_name}}</div>
                        <div class="alt-sim">similarity: ${{item.top1_sim.toFixed(3)}} | cat: ${{item.top1_cat}}</div>
                        <div class="alt-key">Press 1</div>
                    </div>
                    <div class="alt-option" onclick="markWrong(${{item.top2_cat}})">
                        ${{alt2Img}}
                        <div class="alt-name">${{item.top2_name}}</div>
                        <div class="alt-sim">similarity: ${{item.top2_sim.toFixed(3)}} | cat: ${{item.top2_cat}}</div>
                        <div class="alt-key">Press 2</div>
                    </div>
                    <div class="alt-option" onclick="markWrong(${{item.top3_cat}})">
                        ${{alt3Img}}
                        <div class="alt-name">${{item.top3_name}}</div>
                        <div class="alt-sim">similarity: ${{item.top3_sim.toFixed(3)}} | cat: ${{item.top3_cat}}</div>
                        <div class="alt-key">Press 3</div>
                    </div>
                </div>
                <div class="alt-skip">
                    <button onclick="markWrong(null)">I can't tell / None of these (press 0)</button>
                </div>
            </div>
        </div>
    `;
    updateProgress();
    window.scrollTo(0, 0);
}}

function markCorrect() {{
    const item = ALL_ITEMS[currentIndex];
    decisions[item.ann_id] = {{ decision: "correct", new_cat: null }};
    history.push(item.ann_id);
    currentIndex++;
    saveProgress();
    renderCard();
}}

function showAlternatives() {{
    document.getElementById("alternatives-panel").style.display = "block";
}}

function markWrong(newCat) {{
    const item = ALL_ITEMS[currentIndex];
    decisions[item.ann_id] = {{ decision: "wrong", new_cat: newCat, gt_cat: item.gt_cat }};
    history.push(item.ann_id);
    currentIndex++;
    saveProgress();
    renderCard();
}}

function undo() {{
    if (history.length === 0) return;
    const lastId = history.pop();
    delete decisions[lastId];
    // Find the index of this item
    for (let i = 0; i < ALL_ITEMS.length; i++) {{
        if (ALL_ITEMS[i].ann_id === lastId) {{
            currentIndex = i;
            break;
        }}
    }}
    saveProgress();
    renderCard();
}}

function exportResults() {{
    const corrections = Object.entries(decisions)
        .filter(([_, d]) => d.decision === "wrong" && d.new_cat !== null)
        .map(([ann_id, d]) => ({{
            ann_id: parseInt(ann_id),
            old_cat: d.gt_cat,
            new_cat: d.new_cat
        }}));

    const allDecisions = Object.entries(decisions).map(([ann_id, d]) => ({{
        ann_id: parseInt(ann_id),
        decision: d.decision,
        old_cat: d.gt_cat || null,
        new_cat: d.new_cat
    }}));

    const result = {{
        summary: {{
            total_reviewed: Object.keys(decisions).length,
            marked_correct: Object.values(decisions).filter(d => d.decision === "correct").length,
            marked_wrong: Object.values(decisions).filter(d => d.decision === "wrong").length,
            corrections_with_new_cat: corrections.length,
        }},
        corrections: corrections,
        all_decisions: allDecisions,
    }};

    const blob = new Blob([JSON.stringify(result, null, 2)], {{ type: "application/json" }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "annotation_corrections.json";
    a.click();
    URL.revokeObjectURL(url);
}}

// Keyboard shortcuts
document.addEventListener("keydown", (e) => {{
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    const altPanel = document.getElementById("alternatives-panel");
    const altVisible = altPanel && altPanel.style.display === "block";

    if (e.key === "y" || e.key === "Y") {{ markCorrect(); }}
    else if (e.key === "n" || e.key === "N") {{ showAlternatives(); }}
    else if (e.key === "z" || e.key === "Z") {{ undo(); }}
    else if (e.key === "s" || e.key === "S") {{
        currentIndex++;
        saveProgress();
        renderCard();
    }}
    else if (altVisible && e.key === "1") {{ markWrong(ALL_ITEMS[currentIndex].top1_cat); }}
    else if (altVisible && e.key === "2") {{ markWrong(ALL_ITEMS[currentIndex].top2_cat); }}
    else if (altVisible && e.key === "3") {{ markWrong(ALL_ITEMS[currentIndex].top3_cat); }}
    else if (altVisible && e.key === "0") {{ markWrong(null); }}
}});

// Start
renderCard();
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
