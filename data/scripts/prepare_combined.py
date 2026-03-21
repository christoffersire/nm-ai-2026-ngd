"""
Combine real training data with synthetic shelf images into one YOLO dataset.

Usage:
  python data/prepare_combined.py --real datasets/full-class --synthetic datasets/synthetic-shelves --output datasets/combined
  python data/prepare_combined.py --real datasets/full-class --synthetic datasets/synthetic-shelves --output datasets/combined --all-data
"""
import argparse
import shutil
import json
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True, help="Path to real YOLO dataset")
    parser.add_argument("--synthetic", required=True, help="Path to synthetic dataset")
    parser.add_argument("--output", required=True, help="Output combined dataset path")
    parser.add_argument("--all-data", action="store_true", help="Use all-data variant for real")
    args = parser.parse_args()

    real_dir = Path(args.real)
    synth_dir = Path(args.synthetic)
    out_dir = Path(args.output)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    (out_dir / "images" / "train").mkdir(parents=True)
    (out_dir / "labels" / "train").mkdir(parents=True)
    (out_dir / "images" / "val").mkdir(parents=True)
    (out_dir / "labels" / "val").mkdir(parents=True)

    real_count = 0
    synth_count = 0

    # Copy real training images and labels
    real_train_imgs = real_dir / "images" / "train"
    real_train_labels = real_dir / "labels" / "train"

    if real_train_imgs.exists():
        for f in real_train_imgs.iterdir():
            shutil.copy2(f, out_dir / "images" / "train" / f.name)
            real_count += 1

    if real_train_labels.exists():
        for f in real_train_labels.iterdir():
            shutil.copy2(f, out_dir / "labels" / "train" / f.name)

    # Copy synthetic training images and labels
    synth_train_imgs = synth_dir / "images" / "train"
    synth_train_labels = synth_dir / "labels" / "train"

    if synth_train_imgs.exists():
        for f in synth_train_imgs.iterdir():
            shutil.copy2(f, out_dir / "images" / "train" / f.name)
            synth_count += 1

    if synth_train_labels.exists():
        for f in synth_train_labels.iterdir():
            shutil.copy2(f, out_dir / "labels" / "train" / f.name)

    # Copy val from real dataset (unchanged)
    real_val_imgs = real_dir / "images" / "val"
    real_val_labels = real_dir / "labels" / "val"

    val_count = 0
    if real_val_imgs.exists():
        for f in real_val_imgs.iterdir():
            shutil.copy2(f, out_dir / "images" / "val" / f.name)
            val_count += 1

    if real_val_labels.exists():
        for f in real_val_labels.iterdir():
            shutil.copy2(f, out_dir / "labels" / "val" / f.name)

    # Copy data.yaml from real dataset, update path
    yaml_src = real_dir / "data.yaml"
    if yaml_src.exists():
        yaml_content = yaml_src.read_text()
        yaml_content = yaml_content.replace(str(real_dir), str(out_dir))
        (out_dir / "data.yaml").write_text(yaml_content)

    print(f"=== Combined Dataset ===")
    print(f"  Real train images: {real_count}")
    print(f"  Synthetic train images: {synth_count}")
    print(f"  Total train: {real_count + synth_count}")
    print(f"  Val images: {val_count}")
    print(f"  Written to: {out_dir}")


if __name__ == "__main__":
    main()
