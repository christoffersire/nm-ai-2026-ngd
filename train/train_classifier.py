"""
Train YOLOv8 classifier on product crops.

Usage:
  python train/train_classifier.py --dataset datasets/classifier --model yolov8s-cls.pt --epochs 50 --imgsz 224
  python train/train_classifier.py --dataset datasets/classifier --model yolov8m-cls.pt --epochs 50 --imgsz 256
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to classifier dataset (with train/val subdirs)")
    parser.add_argument("--model", default="yolov8s-cls.pt", help="Base model (e.g. yolov8s-cls.pt, yolov8m-cls.pt)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--name", default="classifier", help="Run name")
    parser.add_argument("--device", default="0", help="Device (0 for GPU, cpu for CPU)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()

    model = YOLO(args.model)

    results = model.train(
        data=str(dataset_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        name=args.name,
        device=args.device,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        fliplr=0.5,
        # Training config
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        # Save
        save=True,
        plots=True,
        verbose=True,
    )

    print(f"\nTraining complete. Results saved to: {results.save_dir}")
    print(f"Best model: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
