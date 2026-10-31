import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train enhanced YOLO model for Road Damage Detection.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",
        help="Base model to fine-tune (e.g. yolo11s.pt, yolo11n.pt, yolo11m.pt). Default: yolo11s.pt",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_DIR / "data_2class.yaml"),
        help="Path to data.yaml dataset definition.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs (default: 100).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (default: 15).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image training resolution (default: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8 if torch.cuda.is_available() else 4,
        help="Batch size (default: 8 for GPU, 4 for CPU).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Training device ('0' for CUDA GPU, 'cpu').",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="road_damage_enhanced",
        help="Experiment name under runs/.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_yaml = Path(args.data)

    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    print("==================================================")
    print("      Road Damage Detection - Enhanced Training    ")
    print("==================================================")
    print(f"Base Model : {args.model}")
    print(f"Dataset    : {data_yaml}")
    print(f"Epochs     : {args.epochs} (Patience: {args.patience})")
    print(f"Image Size : {args.imgsz}")
    print(f"Batch Size : {args.batch}")
    print(f"Device     : {args.device}")
    print("Augmentations: HSV jitter, Mosaic, Mixup, Random Rotation/Shear, Copy-Paste")
    print("--------------------------------------------------")

    model = YOLO(args.model)

    # Train with augmentations tailored for asphalt and road hazards
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=42,
        # Enhanced augmentations to boost pothole recall & reduce shadow false-alarms:
        hsv_h=0.015,     # Adjust hue slightly
        hsv_s=0.7,       # Strong saturation variation (dry vs wet roads)
        hsv_v=0.4,       # Strong brightness variation (harsh shadows, overcast)
        degrees=5.0,     # Small rotation for dashcam angle shifts
        translate=0.1,   # Horizontal/vertical shifts
        scale=0.5,       # Scale jitter (distant vs close road defects)
        shear=2.0,       # Perspective shear
        perspective=0.0005, # Road vanishing point perspective simulation
        flipud=0.0,      # Don't flip road upside down
        fliplr=0.5,      # Left-to-right lane flip
        mosaic=1.0,      # 4-image mosaic for small object detection
        mixup=0.15,      # Alpha blend images for occlusion handling
        copy_paste=0.3,  # Pothole/crack instance copy-paste
        project=str(PROJECT_DIR / "runs"),
        name=args.name,
        exist_ok=True,
    )

    print("\nTraining complete!")
    print(f"Best checkpoint: {Path(results.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()

