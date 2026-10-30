
from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent

BEST_MODEL = (
    PROJECT_DIR
    / "runs"
    / "road_damage_baseline"
    / "weights"
    / "best_640.pt"
)

DATA_YAML = PROJECT_DIR / "data_2class.yaml"

if not BEST_MODEL.is_file():
    raise FileNotFoundError(
        f"Trained model not found: {BEST_MODEL}"
    )

model = YOLO(str(BEST_MODEL))

metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
)

print("Test mAP50:", metrics.box.map50)
print("Test mAP50-95:", metrics.box.map)
