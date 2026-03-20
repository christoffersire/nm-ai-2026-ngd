"""
Stage 1: TTA baseline experiments.
Runs multiple TTA configurations and scores each with the hybrid scorer.
"""
import json
import subprocess
import time
from pathlib import Path

# Add parent dir
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorer import hybrid_score

IMAGES_DIR = str(Path.home() / "Downloads" / "train" / "images")
GT_PATH = str(Path.home() / "Downloads" / "train" / "annotations.json")
VAL_SPLIT_PATH = str(Path.home() / "nm-ai-2026-ngd" / "data" / "val_split.json")
CONFIG_PATH = str(Path.home() / "nm-ai-2026-ngd" / "submission" / "config.json")
TTA_SCRIPT = str(Path.home() / "nm-ai-2026-ngd" / "submission" / "run_onnx_tta.py")


def main():
    # Load GT and val split
    with open(GT_PATH) as f:
        gt_data = json.load(f)
    with open(VAL_SPLIT_PATH) as f:
        val_ids = set(json.load(f)["val_ids"])
    gt_val = [g for g in gt_data["annotations"] if g["image_id"] in val_ids]

    configs = [
        ("single-1280",  {"tta_scales": [1280], "tta_flip": False, "fusion_method": "concat_nms"}),
        ("1280+flip",    {"tta_scales": [1280], "tta_flip": True,  "fusion_method": "concat_nms"}),
        ("1024+1280",    {"tta_scales": [1024, 1280], "tta_flip": False, "fusion_method": "concat_nms"}),
        ("1280+1536",    {"tta_scales": [1280, 1536], "tta_flip": False, "fusion_method": "concat_nms"}),
    ]

    base_cfg = {
        "nc": 356, "conf_threshold": 0.03, "iou_nms_threshold": 0.5,
        "max_predictions_per_image": 500, "model_file": "detector.onnx",
        "time_budget": 9999,
    }

    header = f"{'Name':>15} {'det_mAP':>8} {'cls_mAP':>8} {'hybrid':>8} {'delta':>8} {'time':>6} {'preds':>6}"
    print(header)
    print("-" * len(header))

    baseline_hybrid = None

    for name, overrides in configs:
        cfg = {**base_cfg, **overrides}
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f)

        t0 = time.time()
        result = subprocess.run(
            ["python3", TTA_SCRIPT, "--input", IMAGES_DIR, "--output", f"/tmp/tta_{name}.json"],
            capture_output=True, text=True,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"{name:>15} FAILED: {result.stderr[:100]}")
            continue

        with open(f"/tmp/tta_{name}.json") as f:
            preds = json.load(f)
        preds_val = [p for p in preds if p["image_id"] in val_ids]

        r = hybrid_score(preds_val, gt_val)
        h = r["hybrid_score"]

        if baseline_hybrid is None:
            baseline_hybrid = h
            delta = 0.0
        else:
            delta = h - baseline_hybrid

        print(f"{name:>15} {r['detection_mAP']:8.4f} {r['classification_mAP']:8.4f} "
              f"{h:8.4f} {delta:+8.4f} {elapsed:6.1f}s {len(preds_val):>6}")

    print("\nBaseline hybrid:", f"{baseline_hybrid:.4f}" if baseline_hybrid else "N/A")


if __name__ == "__main__":
    main()
