import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

# --------------------------------------------------
# 1. PROJECT DIRECTORY
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO video inference on road damage videos.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(PROJECT_DIR / "best.pt"),
        help="Path to trained model weights (.pt file).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(PROJECT_DIR / "tests"),
        help="Path to a video file or folder containing videos (default: tests/).",
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
        help="Inference frame size (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Device to run on (e.g. '0' for CUDA GPU, 'cpu').",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="video_test_results",
        help="Output folder name inside runs/.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.weights)
    source_path = Path(args.source)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found at: {model_path}\n"
            f"Please specify the path to your trained .pt file with --weights <path>"
        )

    if not source_path.exists():
        raise FileNotFoundError(f"Video source not found at: {source_path}")

    print(f"Loading model: {model_path}")
    print(f"Source: {source_path}")
    print(f"Device: {args.device} | Confidence: {args.conf} | Frame Size: {args.imgsz}")

    model = YOLO(str(model_path))

    model.predict(
        source=str(source_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=0.45,
        device=args.device,
        save=True,
        project=str(PROJECT_DIR / "runs"),
        name=args.name,
        exist_ok=True,
        vid_stride=1,
    )

    out_dir = PROJECT_DIR / "runs" / args.name
    print("\nVideo prediction complete!")
    print(f"Saved annotated videos to: {out_dir}")


if __name__ == "__main__":
    main()
