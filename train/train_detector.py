"""
Train YOLOv8 detector.

Usage:
  python train/train_detector.py --dataset datasets/full-class/data.yaml --model yolov8m.pt --epochs 100 --imgsz 1280
  python train/train_detector.py --dataset datasets/single-class/data.yaml --model yolov8m.pt --epochs 100 --imgsz 1280
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8m.pt", help="Base model (e.g. yolov8m.pt, yolov8l.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 for auto)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--name", default="detector", help="Run name")
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
        # Augmentation — tuned for small dataset with dense shelves
        mosaic=1.0,
        mixup=0.15,
        scale=0.5,
        fliplr=0.5,
        degrees=5.0,
        translate=0.1,
        shear=2.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        erasing=0.4,
        close_mosaic=15,
        # Training config
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        # Save
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    # Print best results
    print(f"\nTraining complete. Results saved to: {results.save_dir}")
    print(f"Best model: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
