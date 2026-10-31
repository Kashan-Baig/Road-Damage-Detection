import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent


def find_best_model():
    candidates = [
        PROJECT_DIR / "best.pt",
        PROJECT_DIR / "runs" / "road_damage_baseline" / "weights" / "best_640.pt",
        PROJECT_DIR / "runs" / "road_damage_2class_640" / "weights" / "best.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return PROJECT_DIR / "best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on the test split.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(find_best_model()),
        help="Path to trained model weights (.pt). Default: best.pt",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_DIR / "data_2class.yaml"),
        help="Path to dataset YAML file.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Evaluation resolution (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation ('0' for GPU, 'cpu').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.weights)
    data_yaml = Path(args.data)

    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found at: {data_yaml}")

    print(f"Evaluating model: {model_path}")
    print(f"Dataset config  : {data_yaml}")
    print(f"Device          : {args.device}")

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=4,
        device=args.device,
        workers=0,
    )

    print("\n---------------- Evaluation Metrics ----------------")
    print(f"Test Precision : {metrics.box.mp:.4f}")
    print(f"Test Recall    : {metrics.box.mr:.4f}")
    print(f"Test mAP50     : {metrics.box.map50:.4f}")
    print(f"Test mAP50-95  : {metrics.box.map:.4f}")
    print("----------------------------------------------------")


if __name__ == "__main__":
    main()
