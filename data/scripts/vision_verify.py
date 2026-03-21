"""
Vision model verification of suspicious annotations.

Sends each suspicious annotation's crop + reference image to 3 vision models
(Claude, GPT, Gemini) and asks if they match. Cross-checks results.

Usage:
  python data/scripts/vision_verify.py
  python data/scripts/vision_verify.py --resume           # Resume from checkpoint
  python data/scripts/vision_verify.py --max 100          # Process first 100 only
"""
import argparse
import base64
import io
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REF_DIR = RAW_DIR / "product_images"
CHECKPOINT_PATH = DATA_DIR / "vision_verify_checkpoint.json"
OUTPUT_PATH = DATA_DIR / "vision_verify_results.json"

# API Keys
ANTHROPIC_KEY = "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"
OPENAI_KEY = "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"
GEMINI_KEY = "AIzaSyBA57XUSyoUOTyq2BIZVQH7IZbmdAbdAww"

PROMPT = """Look at these two images.

IMAGE 1 (left): A crop from a grocery store shelf photo — this is a specific product on the shelf.
IMAGE 2 (right): A reference/official product image for a product called "{product_name}".

Question: Is the product in IMAGE 1 the SAME product as shown in IMAGE 2?

Look carefully at:
- Brand name and logo
- Product name text on the packaging
- Package color, shape, and design
- Size/weight markings

Respond with ONLY a JSON object, nothing else:
{{"match": true}} if they are the same product
{{"match": false, "reason": "<brief reason>"}} if they are different products"""


def img_to_b64(img, max_size=512):
    """Convert PIL Image to base64 JPEG."""
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def parse_response(text):
    """Robustly parse match/mismatch from model response text."""
    raw = text
    text = text.strip()
    # Strip markdown code blocks
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    # Try JSON parse
    try:
        result = json.loads(text)
        return {"match": result.get("match", None), "reason": result.get("reason", ""), "raw": raw}
    except Exception:
        pass
    # Handle truncated JSON like '{"match": false,'
    if '"match": true' in text or '"match":true' in text:
        return {"match": True, "reason": "", "raw": raw}
    if '"match": false' in text or '"match":false' in text:
        return {"match": False, "reason": text, "raw": raw}
    # Fallback: look for keywords
    lower = text.lower()
    if "not the same" in lower or "different product" in lower or "do not match" in lower:
        return {"match": False, "reason": text[:100], "raw": raw}
    if "same product" in lower and "not" not in lower:
        return {"match": True, "reason": "", "raw": raw}
    return {"match": None, "reason": "parse_error", "raw": raw}


def call_claude(crop_b64, ref_b64, product_name):
    """Call Claude Haiku with two images."""
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": crop_b64}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": ref_b64}},
                {"type": "text", "text": PROMPT.format(product_name=product_name)},
            ],
        }],
    )
    return parse_response(msg.content[0].text)


def call_openai(crop_b64, ref_b64, product_name):
    """Call GPT-4o-mini with two images."""
    import openai
    client = openai.OpenAI(api_key=OPENAI_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}", "detail": "auto"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}", "detail": "auto"}},
                {"type": "text", "text": PROMPT.format(product_name=product_name)},
            ],
        }],
    )
    return parse_response(resp.choices[0].message.content)


def call_gemini(crop_b64, ref_b64, product_name):
    """Call Gemini 2.5 Pro with two images."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_KEY)

    crop_bytes = base64.b64decode(crop_b64)
    ref_bytes = base64.b64decode(ref_b64)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=crop_bytes, mime_type="image/jpeg"),
            types.Part.from_bytes(data=ref_bytes, mime_type="image/jpeg"),
            PROMPT.format(product_name=product_name),
        ],
        config=types.GenerateContentConfig(max_output_tokens=300),
    )
    return parse_response(response.text)


def get_ref_image(cat_id, cat_to_code, api_ean_map):
    """Get reference image for a category."""
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
                    return Image.open(p)
                except Exception:
                    continue
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--min-suspicion", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel workers per model")
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

    # Filter candidates
    candidates = [s for s in all_sims if s["suspicion"] >= args.min_suspicion]
    candidates.sort(key=lambda x: -x["suspicion"])
    if args.max:
        candidates = candidates[:args.max]

    # Load checkpoint
    results = {}
    if args.resume and CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            results = {r["ann_id"]: r for r in json.load(f)}
        print(f"Resumed from checkpoint: {len(results)} already processed")

    # Filter out already processed
    to_process = [c for c in candidates if c["ann_id"] not in results]
    print(f"Total candidates: {len(candidates)}, to process: {len(to_process)}")

    # Pre-load and cache images
    img_cache = {}
    t0 = time.time()

    def process_one(cand):
        ann_id = cand["ann_id"]
        ann = ann_by_id[ann_id]
        gt_cat = ann["category_id"]
        gt_name = cat_id_to_name.get(gt_cat, f"Unknown ({gt_cat})")
        img_id = ann["image_id"]

        # Load and crop
        img_info = id_to_img[img_id]
        img_path = RAW_DIR / "images" / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")
        x, y, w, h = ann["bbox"]
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(img.width, int(x + w)), min(img.height, int(y + h))
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        crop = img.crop((x1, y1, x2, y2))
        crop_b64 = img_to_b64(crop)

        # Get reference image
        ref_img = get_ref_image(gt_cat, cat_to_code, api_ean_map)
        if ref_img is None:
            return {
                "ann_id": ann_id, "gt_cat": gt_cat, "gt_name": gt_name,
                "suspicion": cand["suspicion"],
                "claude": {"match": None, "reason": "no_ref"},
                "openai": {"match": None, "reason": "no_ref"},
                "gemini": {"match": None, "reason": "no_ref"},
                "consensus": "no_ref",
            }
        ref_b64 = img_to_b64(ref_img)

        # Call all 3 models IN PARALLEL
        claude_result = openai_result = gemini_result = None

        def _call_claude():
            return call_claude(crop_b64, ref_b64, gt_name)
        def _call_openai():
            return call_openai(crop_b64, ref_b64, gt_name)
        def _call_gemini():
            return call_gemini(crop_b64, ref_b64, gt_name)

        with ThreadPoolExecutor(max_workers=3) as pool:
            fc = pool.submit(_call_claude)
            fo = pool.submit(_call_openai)
            fg = pool.submit(_call_gemini)

            try:
                claude_result = fc.result(timeout=30)
            except Exception as e:
                claude_result = {"match": None, "reason": f"error: {str(e)[:50]}"}
            try:
                openai_result = fo.result(timeout=30)
            except Exception as e:
                openai_result = {"match": None, "reason": f"error: {str(e)[:50]}"}
            try:
                gemini_result = fg.result(timeout=30)
            except Exception as e:
                gemini_result = {"match": None, "reason": f"error: {str(e)[:50]}"}

        # Determine consensus
        votes = []
        for r in [claude_result, openai_result, gemini_result]:
            if r and r["match"] is not None:
                votes.append(r["match"])

        if len(votes) >= 2:
            true_votes = sum(1 for v in votes if v)
            false_votes = sum(1 for v in votes if not v)
            if true_votes >= 2:
                consensus = "match"
            elif false_votes >= 2:
                consensus = "mismatch"
            else:
                consensus = "disagree"
        elif len(votes) == 1:
            consensus = "match" if votes[0] else "mismatch"
        else:
            consensus = "error"

        return {
            "ann_id": ann_id,
            "gt_cat": gt_cat,
            "gt_name": gt_name,
            "suspicion": cand["suspicion"],
            "claude": {k: v for k, v in claude_result.items() if k != "raw"},
            "openai": {k: v for k, v in openai_result.items() if k != "raw"},
            "gemini": {k: v for k, v in gemini_result.items() if k != "raw"},
            "consensus": consensus,
        }

    # Process with progress tracking
    processed = 0
    batch_size = 10  # checkpoint every N items

    for i in range(0, len(to_process), batch_size):
        batch = to_process[i:i + batch_size]

        for cand in batch:
            result = process_one(cand)
            if result:
                results[result["ann_id"]] = result
                processed += 1

                c = result["claude"]["match"]
                o = result["openai"]["match"]
                g = result["gemini"]["match"]
                icon = {"match": "✓", "mismatch": "✗", "disagree": "?", "error": "E", "no_ref": "-"}.get(result["consensus"], "?")

                c_s = "Y" if c else ("N" if c is False else "?")
                o_s = "Y" if o else ("N" if o is False else "?")
                g_s = "Y" if g else ("N" if g is False else "?")

                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(to_process) - i - len(batch)) / rate if rate > 0 else 0

                print(f"  [{len(results)}/{len(candidates)}] {icon} ann {result['ann_id']:>5} "
                      f"C:{c_s} O:{o_s} G:{g_s} = {result['consensus']:8s} "
                      f"({rate:.1f}/s, ETA:{eta/60:.0f}m) "
                      f"{result['gt_name'][:35]}")

            # Small delay to respect rate limits
            time.sleep(0.15)

        # Checkpoint
        checkpoint_data = list(results.values())
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoint_data, f)

    # Final output
    all_results = list(results.values())

    from collections import Counter
    consensus_counts = Counter(r["consensus"] for r in all_results)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total processed: {len(all_results)}")
    for c, n in consensus_counts.most_common():
        print(f"  {c:15s}: {n:>5} ({100*n/len(all_results):.1f}%)")

    # Disagreements detail
    disagree = [r for r in all_results if r["consensus"] == "disagree"]
    if disagree:
        print(f"\nDisagreements ({len(disagree)}):")
        for r in disagree[:20]:
            c = "Y" if r["claude"]["match"] else "N"
            o = "Y" if r["openai"]["match"] else "N"
            g = "Y" if r["gemini"]["match"] else "N"
            print(f"  ann {r['ann_id']:>5}: C:{c} O:{o} G:{g} — {r['gt_name'][:40]}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "summary": dict(consensus_counts),
            "total": len(all_results),
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
