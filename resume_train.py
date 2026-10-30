from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent

LAST_CHECKPOINT = (
    PROJECT_DIR
    / "runs"
    / "road_damage_baseline"
    / "weights"
    / "last.pt"
)

model = YOLO(str(LAST_CHECKPOINT))

results = model.train(
    resume=True
)

print("Training resumed and finished.")