import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO prediction on a test image or folder.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(PROJECT_DIR / "best.pt"),
        help="Path to trained model weights (.pt). Default: best.pt",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(PROJECT_DIR / "road_damage_2class" / "images" / "test"),
        help="Path to image file or directory of images.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Inference device ('0' for GPU, 'cpu').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.weights)
    source_path = Path(args.source)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not source_path.exists():
        raise FileNotFoundError(f"Source image or folder not found: {source_path}")

    print(f"Loading model: {model_path}")
    print(f"Predicting on: {source_path}")

    model = YOLO(str(model_path))

    results = model.predict(
        source=str(source_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=0.45,
        device=args.device,
        save=True,
        project=str(PROJECT_DIR / "runs"),
        name=f"predictions_conf_{str(args.conf).replace('.', '')}",
        exist_ok=True,
    )

    print("\nPrediction complete!")
    print(f"Results saved inside: {PROJECT_DIR / 'runs'}")


if __name__ == "__main__":
    main()
