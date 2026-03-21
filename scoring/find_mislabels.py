"""
Find likely mislabeled annotations by comparing detector predictions to GT.
Where the detector is very confident about a DIFFERENT class than GT, it may indicate a labeling error.

Usage: python scoring/find_mislabels.py
"""
import json
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO


def compute_iou(gt_bbox, pred_xyxy):
    gt_x, gt_y, gt_w, gt_h = gt_bbox
    px1, py1, px2, py2 = pred_xyxy
    pw, ph = px2 - px1, py2 - py1

    ix = max(gt_x, px1)
    iy = max(gt_y, py1)
    ix2 = min(gt_x + gt_w, px2)
    iy2 = min(gt_y + gt_h, py2)
    inter = max(0, ix2 - ix) * max(0, iy2 - iy)
    union = gt_w * gt_h + pw * ph - inter
    return inter / union if union > 0 else 0


def main():
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_DIR / "data"
    RAW_DIR = DATA_DIR / "raw"

    model = YOLO("runs/detect/det-11x-v2-alldata/weights/best.pt")

    ann_path = RAW_DIR / "annotations.json"
    with open(ann_path) as f:
        data = json.load(f)

    cat_names = {c["id"]: c["name"] for c in data["categories"]}
    id_to_img = {img["id"]: img for img in data["images"]}

    img_anns = defaultdict(list)
    for ann in data["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    img_dir = RAW_DIR / "images"

    mismatches = []
    total_checked = 0

    for img_id, anns in sorted(img_anns.items()):
        img_info = id_to_img[img_id]
        fname = img_info["file_name"]
        fpath = img_dir / fname
        if not fpath.exists():
            continue

        try:
            results = model.predict(str(fpath), conf=0.25, iou=0.5, max_det=500, verbose=False, device=0)
        except Exception:
            continue

        for r in results:
            if r.boxes is None:
                continue
            pred_boxes = r.boxes.xyxy.cpu().tolist()
            pred_cls = r.boxes.cls.cpu().tolist()
            pred_conf = r.boxes.conf.cpu().tolist()

            for ann in anns:
                gt_cat = ann["category_id"]
                total_checked += 1

                best_iou = 0
                best_pred_cls = -1
                best_pred_conf = 0

                for pb, pc, pconf in zip(pred_boxes, pred_cls, pred_conf):
                    iou = compute_iou(ann["bbox"], pb)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_cls = int(pc)
                        best_pred_conf = pconf

                if best_iou > 0.5 and best_pred_cls != gt_cat and best_pred_conf > 0.5:
                    mismatches.append({
                        "image": fname,
                        "ann_id": ann["id"],
                        "gt_cat": gt_cat,
                        "gt_name": cat_names.get(gt_cat, "?"),
                        "pred_cat": best_pred_cls,
                        "pred_name": cat_names.get(best_pred_cls, "?"),
                        "pred_conf": best_pred_conf,
                        "iou": best_iou,
                    })

    mismatches.sort(key=lambda x: -x["pred_conf"])

    print(f"Checked {total_checked} annotations")
    print(f"Potential mislabels: {len(mismatches)}")
    print()
    print(f"Top 40 most likely labeling errors:")
    print(f"{'Conf':>5} {'IoU':>5}  {'GT Cat':>6} {'GT Name':>42}  ->  {'Pred Cat':>8} Pred Name [Image]")
    print("-" * 130)
    for m in mismatches[:40]:
        gt = f"cat{m['gt_cat']:>3} {m['gt_name'][:38]:>38}"
        pr = f"cat{m['pred_cat']:>3} {m['pred_name'][:38]}"
        print(f"{m['pred_conf']:5.2f} {m['iou']:5.2f}  {gt}  ->  {pr}  [{m['image']}]")

    # Save full results
    with open("/tmp/mislabel_candidates.json", "w") as f:
        json.dump(mismatches, f, indent=2)
    print(f"\nFull results saved to /tmp/mislabel_candidates.json")


if __name__ == "__main__":
    main()
