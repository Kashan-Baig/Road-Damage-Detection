import argparse
from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Export trained YOLO weights to ONNX format.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(PROJECT_DIR / "best.pt"),
        help="Path to trained .pt weights file (default: best.pt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference frame resolution (default: 640).",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 half-precision for faster GPU inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    print(f"Loading weights from: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"Exporting to ONNX format (imgsz={args.imgsz}, half={args.half})...")
    exported_file = model.export(
        format="onnx",
        imgsz=args.imgsz,
        half=args.half,
        dynamic=True,
        simplify=True,
    )

    print("\nExport successful!")
    print(f"ONNX Model saved to: {exported_file}")


if __name__ == "__main__":
    main()

