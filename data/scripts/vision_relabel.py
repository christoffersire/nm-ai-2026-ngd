"""
Vision-based relabeling: for annotations confirmed as mislabeled,
determine the CORRECT category using 3 vision models.

Sends each mislabeled crop + top N candidate reference images to
Claude, GPT, and Gemini, asks which product it is.

Usage:
  python data/scripts/vision_relabel.py                    # Process all mismatches
  python data/scripts/vision_relabel.py --resume           # Resume from checkpoint
  python data/scripts/vision_relabel.py --max 50           # Process first 50 only
"""
import argparse
import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REF_DIR = RAW_DIR / "product_images"
VERIFY_RESULTS = DATA_DIR / "vision_verify_results.json"
SIMILARITIES = DATA_DIR / "all_similarities.json"
CHECKPOINT_PATH = DATA_DIR / "vision_relabel_checkpoint.json"
OUTPUT_PATH = DATA_DIR / "vision_relabel_results.json"

ANTHROPIC_KEY = "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"
OPENAI_KEY = "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"
GEMINI_KEY = "AIzaSyBA57XUSyoUOTyq2BIZVQH7IZbmdAbdAww"

NUM_CANDIDATES = 8  # Number of candidate products to show


def img_to_b64(img, max_size=400):
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_prompt(candidates):
    """Build the identification prompt with candidate list."""
    candidate_list = "\n".join(
        f"  OPTION {i+1}: {c['name']} (category_id: {c['cat_id']})"
        for i, c in enumerate(candidates)
    )
    return f"""Look at IMAGE 1 — it is a crop from a grocery store shelf photo showing ONE product.

The remaining images (IMAGE 2 through IMAGE {len(candidates)+1}) are reference/official product images for these candidate products:

{candidate_list}

The reference images are shown in the SAME ORDER as the options above.

Which option matches the product in IMAGE 1? Look carefully at:
- Brand name and logo text
- Product name text on the packaging
- Package color, shape, and design

Respond with ONLY a JSON object:
{{"choice": <option number 1-{len(candidates)}>, "cat_id": <category_id>, "confidence": "high"|"medium"|"low", "reason": "<brief reason>"}}

If NONE of the options match, respond with:
{{"choice": 0, "cat_id": null, "confidence": "low", "reason": "<what product you think it is>"}}"""


def parse_relabel_response(text):
    """Parse the relabeling response."""
    raw = text.strip()
    text = raw
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        result = json.loads(text)
        return {
            "choice": result.get("choice", 0),
            "cat_id": result.get("cat_id"),
            "confidence": result.get("confidence", "unknown"),
            "reason": result.get("reason", ""),
        }
    except Exception:
        # Try to extract choice number
        import re
        m = re.search(r'"choice"\s*:\s*(\d+)', raw)
        cat_m = re.search(r'"cat_id"\s*:\s*(\d+)', raw)
        if m:
            return {
                "choice": int(m.group(1)),
                "cat_id": int(cat_m.group(1)) if cat_m else None,
                "confidence": "unknown",
                "reason": "partial_parse",
            }
        return {"choice": 0, "cat_id": None, "confidence": "error", "reason": f"parse_error: {raw[:80]}"}


def call_claude_relabel(crop_b64, candidate_b64s, prompt):
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": crop_b64}}]
    for b64 in candidate_b64s:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
    content.append({"type": "text", "text": prompt})

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": content}],
    )
    return parse_relabel_response(msg.content[0].text)


def call_openai_relabel(crop_b64, candidate_b64s, prompt):
    import openai
    client = openai.OpenAI(api_key=OPENAI_KEY)
    content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}", "detail": "auto"}}]
    for b64 in candidate_b64s:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "auto"}})
    content.append({"type": "text", "text": prompt})

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        messages=[{"role": "user", "content": content}],
    )
    return parse_relabel_response(resp.choices[0].message.content)


def call_gemini_relabel(crop_b64, candidate_b64s, prompt, candidates=None):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_KEY)

    parts = [types.Part.from_bytes(data=base64.b64decode(crop_b64), mime_type="image/jpeg")]
    for b64 in candidate_b64s:
        parts.append(types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/jpeg"))
    parts.append(prompt)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=parts,
        config=types.GenerateContentConfig(max_output_tokens=300),
    )
    result = parse_relabel_response(response.text)
    # If choice was parsed but cat_id is missing, look it up from candidates
    if result["cat_id"] is None and result["choice"] > 0 and candidates:
        choice_idx = result["choice"] - 1
        if 0 <= choice_idx < len(candidates):
            result["cat_id"] = candidates[choice_idx]["cat_id"]
    return result


def get_ref_b64(cat_id, cat_to_code, api_ean_map, max_size=250):
    """Get base64 reference image for a category."""
    codes = []
    if cat_id in cat_to_code:
        codes.append(cat_to_code[cat_id])
    if cat_id in api_ean_map:
        codes.append(api_ean_map[cat_id])
    for code in codes:
        for name in ["main.jpg", "front.jpg", "back.jpg"]:
            p = REF_DIR / code / name
            if p.exists():
                try:
                    return img_to_b64(Image.open(p), max_size=max_size)
                except Exception:
                    continue
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    with open(RAW_DIR / "annotations.json") as f:
        data = json.load(f)
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    with open(SIMILARITIES) as f:
        all_sims = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    cat_name_to_id = {c["name"]: c["id"] for c in data["categories"]}
    id_to_img = {img["id"]: img for img in data["images"]}
    ann_by_id = {a["id"]: a for a in data["annotations"]}
    sim_by_ann = {s["ann_id"]: s for s in all_sims}

    cat_to_code = {}
    for p in meta["products"]:
        if p["product_name"] in cat_name_to_id:
            cat_to_code[cat_name_to_id[p["product_name"]]] = p["product_code"]

    api_ean_map = {}
    verif_path = DATA_DIR / "product_verification.json"
    if verif_path.exists():
        with open(verif_path) as f:
            verif = json.load(f)
        for r in verif.get("results", []):
            if r.get("api_ean") and r["cat_id"] not in cat_to_code:
                api_ean_map[r["cat_id"]] = r["api_ean"]

    # Load verification results — get mismatches
    if not VERIFY_RESULTS.exists():
        # Fall back to checkpoint
        cp = DATA_DIR / "vision_verify_checkpoint.json"
        if not cp.exists():
            print("No verification results yet. Run vision_verify.py first.")
            return
        with open(cp) as f:
            verify_data = json.load(f)
    else:
        with open(VERIFY_RESULTS) as f:
            verify_data = json.load(f).get("results", [])

    mismatches = [r for r in verify_data if r["consensus"] == "mismatch"]
    print(f"Found {len(mismatches)} confirmed mismatches to relabel")

    if args.max:
        mismatches = mismatches[:args.max]

    # Pre-cache reference images for all categories
    print("Caching reference images...")
    ref_b64_cache = {}
    for cat_id in cat_id_to_name:
        b64 = get_ref_b64(cat_id, cat_to_code, api_ean_map)
        if b64:
            ref_b64_cache[cat_id] = b64
    print(f"  Cached {len(ref_b64_cache)} reference images")

    # Load checkpoint
    results = {}
    if args.resume and CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            results = {r["ann_id"]: r for r in json.load(f)}
        print(f"Resumed: {len(results)} already processed")

    to_process = [m for m in mismatches if m["ann_id"] not in results]
    print(f"To process: {len(to_process)}")

    t0 = time.time()

    for idx, mismatch in enumerate(to_process):
        ann_id = mismatch["ann_id"]
        ann = ann_by_id[ann_id]
        gt_cat = ann["category_id"]
        gt_name = cat_id_to_name.get(gt_cat, "?")
        sim_data = sim_by_ann.get(ann_id, {})

        # Get crop
        img_info = id_to_img[ann["image_id"]]
        img_path = RAW_DIR / "images" / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")
        x, y, w, h = ann["bbox"]
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(img.width, int(x + w)), min(img.height, int(y + h))
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        crop = img.crop((x1, y1, x2, y2))
        crop_b64 = img_to_b64(crop)

        # Build candidate list from embedding similarities
        # Include top3 from embeddings + nearby categories on the same shelf
        candidate_cats = set()

        # Add top matches from embedding
        for key in ["top1_cat", "top2_cat", "top3_cat"]:
            if key in sim_data:
                candidate_cats.add(sim_data[key])

        # Add categories that appear in the same image (spatial context)
        img_anns = [a for a in data["annotations"] if a["image_id"] == ann["image_id"]]
        img_cat_counts = {}
        for a in img_anns:
            c = a["category_id"]
            if c != gt_cat:
                img_cat_counts[c] = img_cat_counts.get(c, 0) + 1
        # Add top co-occurring categories
        co_occurring = sorted(img_cat_counts.items(), key=lambda x: -x[1])
        for c, _ in co_occurring[:5]:
            candidate_cats.add(c)

        # Remove GT category and categories without reference images
        candidate_cats.discard(gt_cat)
        candidate_cats = [c for c in candidate_cats if c in ref_b64_cache]

        # Limit to NUM_CANDIDATES
        # Prioritize: top embedding matches first, then co-occurring
        ordered_candidates = []
        for key in ["top1_cat", "top2_cat", "top3_cat"]:
            c = sim_data.get(key)
            if c and c in candidate_cats and c not in [x["cat_id"] for x in ordered_candidates]:
                ordered_candidates.append({"cat_id": c, "name": cat_id_to_name.get(c, "?")})
        for c, _ in co_occurring:
            if c in candidate_cats and c not in [x["cat_id"] for x in ordered_candidates]:
                ordered_candidates.append({"cat_id": c, "name": cat_id_to_name.get(c, "?")})
        ordered_candidates = ordered_candidates[:NUM_CANDIDATES]

        if not ordered_candidates:
            results[ann_id] = {
                "ann_id": ann_id, "gt_cat": gt_cat, "gt_name": gt_name,
                "new_cat": None, "consensus": "no_candidates",
                "claude": {}, "openai": {}, "gemini": {},
            }
            continue

        # Get reference images for candidates
        candidate_b64s = [ref_b64_cache[c["cat_id"]] for c in ordered_candidates]
        prompt = build_prompt(ordered_candidates)

        # Call all 3 models in parallel
        claude_r = openai_r = gemini_r = None
        with ThreadPoolExecutor(max_workers=3) as pool:
            fc = pool.submit(call_claude_relabel, crop_b64, candidate_b64s, prompt)
            fo = pool.submit(call_openai_relabel, crop_b64, candidate_b64s, prompt)
            fg = pool.submit(call_gemini_relabel, crop_b64, candidate_b64s, prompt, ordered_candidates)

            try:
                claude_r = fc.result(timeout=45)
            except Exception as e:
                claude_r = {"choice": 0, "cat_id": None, "confidence": "error", "reason": str(e)[:50]}
            try:
                openai_r = fo.result(timeout=45)
            except Exception as e:
                openai_r = {"choice": 0, "cat_id": None, "confidence": "error", "reason": str(e)[:50]}
            try:
                gemini_r = fg.result(timeout=45)
            except Exception as e:
                gemini_r = {"choice": 0, "cat_id": None, "confidence": "error", "reason": str(e)[:50]}

        # Resolve cat_id from choice for any model that returned choice but not cat_id
        for model_r in [claude_r, openai_r, gemini_r]:
            if model_r.get("cat_id") is None and model_r.get("choice", 0) > 0:
                choice_idx = model_r["choice"] - 1
                if 0 <= choice_idx < len(ordered_candidates):
                    model_r["cat_id"] = ordered_candidates[choice_idx]["cat_id"]

        # Determine consensus
        votes = {}
        for model_r in [claude_r, openai_r, gemini_r]:
            cat = model_r.get("cat_id")
            if cat is not None and cat != 0:
                votes[cat] = votes.get(cat, 0) + 1

        if votes:
            best_cat = max(votes, key=votes.get)
            vote_count = votes[best_cat]
            if vote_count >= 2:
                consensus = "agree"
            else:
                consensus = "split"
        else:
            best_cat = None
            consensus = "none"

        result = {
            "ann_id": ann_id,
            "gt_cat": gt_cat,
            "gt_name": gt_name,
            "new_cat": best_cat if consensus == "agree" else None,
            "new_name": cat_id_to_name.get(best_cat, "?") if best_cat else None,
            "consensus": consensus,
            "vote_count": votes.get(best_cat, 0) if best_cat else 0,
            "claude": claude_r,
            "openai": openai_r,
            "gemini": gemini_r,
            "candidates": [c["cat_id"] for c in ordered_candidates],
        }
        results[ann_id] = result

        c_cat = claude_r.get("cat_id", "?")
        o_cat = openai_r.get("cat_id", "?")
        g_cat = gemini_r.get("cat_id", "?")
        icon = {"agree": "✓", "split": "?", "none": "✗"}.get(consensus, "?")
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        eta = (len(to_process) - idx - 1) / rate if rate > 0 else 0

        new_info = f"-> {best_cat}({cat_id_to_name.get(best_cat, '?')[:20]})" if best_cat else "-> ???"
        print(f"  [{len(results)}/{len(mismatches)}] {icon} ann {ann_id:>5}: "
              f"C:{c_cat} O:{o_cat} G:{g_cat} {consensus:5s} "
              f"{gt_name[:25]} {new_info} "
              f"({rate:.1f}/s, ETA:{eta/60:.0f}m)")

        time.sleep(0.15)

        # Checkpoint every 10
        if (idx + 1) % 10 == 0:
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(list(results.values()), f)

    # Final save
    all_results = list(results.values())
    from collections import Counter
    consensus_counts = Counter(r["consensus"] for r in all_results)

    print(f"\n{'='*60}")
    print("RELABELING RESULTS")
    print(f"{'='*60}")
    print(f"Total: {len(all_results)}")
    for c, n in consensus_counts.most_common():
        print(f"  {c:15s}: {n}")

    agreed = [r for r in all_results if r["consensus"] == "agree"]
    print(f"\nConfident relabels (2+ models agree): {len(agreed)}")
    for r in agreed[:20]:
        print(f"  ann {r['ann_id']:>5}: cat {r['gt_cat']} ({r['gt_name'][:25]}) "
              f"-> cat {r['new_cat']} ({r['new_name'][:25]})")

    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "summary": dict(consensus_counts),
            "total": len(all_results),
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {OUTPUT_PATH}")

    # Also save a clean corrections file ready to apply
    corrections = [
        {"ann_id": r["ann_id"], "old_cat": r["gt_cat"], "new_cat": r["new_cat"]}
        for r in agreed if r["new_cat"] is not None
    ]
    corrections_path = DATA_DIR / "auto_corrections.json"
    with open(corrections_path, "w") as f:
        json.dump(corrections, f, indent=2)
    print(f"Ready-to-apply corrections: {len(corrections)} -> {corrections_path}")


if __name__ == "__main__":
    main()
