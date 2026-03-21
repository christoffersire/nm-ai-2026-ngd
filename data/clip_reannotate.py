"""
CLIP-based annotation re-classification.

Builds visual embeddings for every product category using reference images,
then re-classifies every annotation crop against those embeddings.

High-confidence matches (cosine similarity > --threshold) are reassigned.
Low-confidence crops keep their original category_id and are logged for review.
Cat#355 (unknown_product) crops are always re-assigned to the best match.

Output:
  ~/Downloads/train/annotations_clip.json   — corrected COCO file
  data/audit_out/clip_changes.csv           — what was changed + confidence
  data/audit_out/clip_unmatched.csv         — low-confidence crops for review
  data/audit_out/clip_embeddings.pt         — cached category embeddings (fast rerun)

Usage:
  pip install open-clip-torch torch
  python3 data/clip_reannotate.py
  python3 data/clip_reannotate.py --model ViT-L-14 --threshold 0.80
  python3 data/clip_reannotate.py --dry-run
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ANNOTATIONS_PATH  = Path.home() / "Downloads" / "train" / "annotations.json"
ANNOTATIONS_OUT   = Path.home() / "Downloads" / "train" / "annotations_clip.json"
IMAGES_DIR        = Path.home() / "Downloads" / "train" / "images"
PRODUCT_IMAGES    = Path.home() / "Downloads" / "NM_NGD_product_images"
CATEGORY_MAPPING  = Path(__file__).parent / "category_mapping.json"
OUT_DIR           = Path(__file__).parent / "audit_out"
EMBED_CACHE       = OUT_DIR / "clip_embeddings.pt"

UNKNOWN_CAT_ID    = 355
CROP_PADDING      = 8
MIN_CROP_PX       = 20
REF_PRIORITY      = ["front.jpg", "main.jpg", "back.jpg", "left.jpg", "right.jpg", "top.jpg"]

DEFAULT_MODEL     = "ViT-B-32"
DEFAULT_PRETRAIN  = "openai"
DEFAULT_THRESHOLD = 0.77


def get_ref_images(product_code: str) -> list[Path]:
    if not product_code:
        return []
    folder = PRODUCT_IMAGES / product_code
    if not folder.exists():
        return []
    ordered = []
    for name in REF_PRIORITY:
        p = folder / name
        if p.exists():
            ordered.append(p)
    for p in sorted(folder.glob("*.jpg")):
        if p not in ordered:
            ordered.append(p)
    return ordered


def crop_annotation(shelf_img, bbox):
    x, y, w, h = [int(v) for v in bbox]
    iw, ih = shelf_img.size
    x1 = max(0, x - CROP_PADDING)
    y1 = max(0, y - CROP_PADDING)
    x2 = min(iw, x + w + CROP_PADDING)
    y2 = min(ih, y + h + CROP_PADDING)
    if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
        return None
    return shelf_img.crop((x1, y1, x2, y2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=DEFAULT_MODEL,
                        help=f"CLIP model (default: {DEFAULT_MODEL})")
    parser.add_argument("--pretrain",   default=DEFAULT_PRETRAIN,
                        help=f"Pretrained weights (default: {DEFAULT_PRETRAIN})")
    parser.add_argument("--threshold",  type=float, default=DEFAULT_THRESHOLD,
                        help=f"Cosine similarity threshold for auto-reassignment (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--no-cache",   action="store_true",
                        help="Ignore cached embeddings and recompute")
    args = parser.parse_args()

    try:
        import torch
        import open_clip
        from PIL import Image
    except ImportError:
        raise SystemExit("Kjør: pip install open-clip-torch torch Pillow")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model:  {args.model} / {args.pretrain}")
    print(f"Threshold: {args.threshold}")

    # ── Load CLIP ──────────────────────────────────────────────────────────────
    print("\nLaster CLIP-modell …")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrain, device=device
    )
    model.eval()

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Laster annotations og mapping …")
    data    = json.load(open(ANNOTATIONS_PATH))
    cats    = {c["id"]: c["name"] for c in data["categories"]}
    img_map = {i["id"]: i["file_name"] for i in data["images"]}
    mapping = {}
    if CATEGORY_MAPPING.exists():
        mapping = {int(k): v for k, v in json.load(open(CATEGORY_MAPPING)).items()}

    ann_by_cat = defaultdict(list)
    for a in data["annotations"]:
        ann_by_cat[a["category_id"]].append(a)

    # ── Build category embeddings ──────────────────────────────────────────────
    # Check cache
    cat_embeddings = {}   # cat_id → tensor (embed_dim,)
    cats_with_refs = []
    cats_no_refs   = []

    if EMBED_CACHE.exists() and not args.no_cache:
        print(f"Laster cached embeddings fra {EMBED_CACHE} …")
        cached = torch.load(EMBED_CACHE, map_location=device)
        cat_embeddings = cached
        print(f"  {len(cat_embeddings)} kategorier lastet fra cache")
    else:
        print("\nBygger kategori-embeddings fra referansebilder …")
        for cid in sorted(cats.keys()):
            if cid == UNKNOWN_CAT_ID:
                continue
            info = mapping.get(cid, {})
            code = info.get("product_code")
            ref_paths = get_ref_images(code)

            if not ref_paths:
                cats_no_refs.append(cid)
                continue

            # Encode all reference images for this category
            imgs_tensor = []
            for p in ref_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    imgs_tensor.append(preprocess(img))
                except Exception:
                    continue

            if not imgs_tensor:
                cats_no_refs.append(cid)
                continue

            cats_with_refs.append(cid)
            batch = torch.stack(imgs_tensor).to(device)
            with torch.no_grad():
                feats = model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                cat_embeddings[cid] = feats.mean(dim=0)  # average embedding

            if len(cats_with_refs) % 50 == 0:
                print(f"  {len(cats_with_refs)} kategorier …")

        # Also encode category names as text (helps for cats without ref images)
        print("Bygger tekst-embeddings for kategorier uten referansebilder …")
        tokenizer = open_clip.get_tokenizer(args.model)
        for cid in cats_no_refs:
            name = cats.get(cid, "")
            if not name:
                continue
            tokens = tokenizer([f"a photo of {name}", name]).to(device)
            with torch.no_grad():
                feats = model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                cat_embeddings[cid] = feats.mean(dim=0)

        # Save cache
        OUT_DIR.mkdir(exist_ok=True)
        torch.save(cat_embeddings, EMBED_CACHE)
        print(f"Embeddings cachet → {EMBED_CACHE}")
        print(f"  Med referansebilder: {len(cats_with_refs)}")
        print(f"  Kun tekst:           {len(cats_no_refs)}")

    # Stack all category embeddings into a matrix for fast similarity search
    cat_ids_list  = sorted(cat_embeddings.keys())
    embed_matrix  = torch.stack([cat_embeddings[c] for c in cat_ids_list])  # (N, D)
    embed_matrix  = embed_matrix / embed_matrix.norm(dim=-1, keepdim=True)

    # ── Re-classify annotations ────────────────────────────────────────────────
    print(f"\nRe-klassifiserer {len(data['annotations'])} annotations …")

    changes   = []   # (ann_id, old_cat, new_cat, sim, old_name, new_name)
    unmatched = []   # (ann_id, old_cat, best_cat, sim)

    new_annotations = []
    shelf_cache     = {}  # img_id → PIL Image

    total   = len(data["annotations"])
    changed = 0
    kept    = 0
    skipped = 0

    for i, ann in enumerate(data["annotations"]):
        if i % 1000 == 0:
            print(f"  {i}/{total} … (endret={changed}, beholdt={kept}, hoppet={skipped})")

        old_cat = ann["category_id"]

        # Load shelf image (cached)
        img_id = ann["image_id"]
        if img_id not in shelf_cache:
            fname = img_map.get(img_id)
            if not fname:
                new_annotations.append(ann)
                skipped += 1
                continue
            try:
                shelf_cache[img_id] = Image.open(IMAGES_DIR / fname).convert("RGB")
            except Exception:
                new_annotations.append(ann)
                skipped += 1
                continue

        # Crop
        crop = crop_annotation(shelf_cache[img_id], ann["bbox"])
        if crop is None:
            new_annotations.append(ann)
            skipped += 1
            continue

        # Encode crop
        crop_tensor = preprocess(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            crop_feat = model.encode_image(crop_tensor)
            crop_feat = crop_feat / crop_feat.norm(dim=-1, keepdim=True)

        # Cosine similarity against all categories
        sims = (embed_matrix @ crop_feat.T).squeeze(-1)  # (N,)
        best_idx  = sims.argmax().item()
        best_sim  = sims[best_idx].item()
        best_cat  = cat_ids_list[best_idx]

        # Decision
        is_unknown = (old_cat == UNKNOWN_CAT_ID)
        reassign   = (best_sim >= args.threshold and best_cat != old_cat) or is_unknown

        if reassign:
            new_ann = dict(ann)
            new_ann["category_id"] = best_cat
            new_annotations.append(new_ann)
            changes.append({
                "ann_id":   ann["id"],
                "image_id": img_id,
                "old_cat":  old_cat,
                "new_cat":  best_cat,
                "similarity": round(best_sim, 4),
                "old_name": cats.get(old_cat, ""),
                "new_name": cats.get(best_cat, ""),
                "was_unknown": is_unknown,
            })
            changed += 1
        else:
            new_annotations.append(ann)
            kept += 1
            if best_sim < args.threshold:
                unmatched.append({
                    "ann_id":     ann["id"],
                    "image_id":   img_id,
                    "old_cat":    old_cat,
                    "old_name":   cats.get(old_cat, ""),
                    "best_cat":   best_cat,
                    "best_name":  cats.get(best_cat, ""),
                    "similarity": round(best_sim, 4),
                })

    print(f"\nFerdig:")
    print(f"  Endret:   {changed}")
    print(f"  Beholdt:  {kept}")
    print(f"  Hoppet:   {skipped}")
    pct = changed / total * 100
    print(f"  Endret %: {pct:.1f}%")

    from_unknown = sum(1 for c in changes if c["was_unknown"])
    reassigned   = sum(1 for c in changes if not c["was_unknown"])
    print(f"\n  Fra cat#355 (unknown): {from_unknown}")
    print(f"  Reklassifisert:        {reassigned}")

    if args.dry_run:
        print("\n[DRY RUN] Ingen filer skrevet.")
        return

    # ── Write outputs ──────────────────────────────────────────────────────────
    out_data = dict(data)
    out_data["annotations"] = new_annotations
    ANNOTATIONS_OUT.write_text(
        json.dumps(out_data, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nNy annotations → {ANNOTATIONS_OUT}")

    # changes CSV
    changes_path = OUT_DIR / "clip_changes.csv"
    with open(changes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ann_id", "image_id", "old_cat", "new_cat",
            "similarity", "old_name", "new_name", "was_unknown"
        ])
        writer.writeheader()
        writer.writerows(changes)
    print(f"Endringer       → {changes_path}  ({len(changes)} rader)")

    # unmatched CSV
    unmatched_path = OUT_DIR / "clip_unmatched.csv"
    with open(unmatched_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ann_id", "image_id", "old_cat", "old_name",
            "best_cat", "best_name", "similarity"
        ])
        writer.writeheader()
        writer.writerows(unmatched)
    print(f"Lav konfidens   → {unmatched_path}  ({len(unmatched)} rader)")

    print(f"\nDin teammate bruker: {ANNOTATIONS_OUT}")


if __name__ == "__main__":
    main()
