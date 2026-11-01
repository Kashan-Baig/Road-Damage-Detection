from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
DATA_YAML = PROJECT_DIR / "data_2class.yaml"

if not DATA_YAML.is_file():
    raise FileNotFoundError(
        f"{DATA_YAML} not found. Run prepare_2class_dataset.py first."
    )

# Start from the original pretrained YOLO weights. Do NOT use the old
# 3-class best.pt as the starting checkpoint for this first clean comparison.
model = YOLO("yolo11n.pt")

results = model.train(
    data=str(DATA_YAML),
    epochs=100,          # Maximum epochs; early stopping may stop sooner
    patience=10,         # Stop if validation fitness does not improve for 10 epochs
    imgsz=640,           # Higher resolution for small road damage
    batch=4,             # If CPU training is too slow, try batch=2
    device="cpu",
    workers=0,
    seed=42,
    project=str(PROJECT_DIR / "runs"),
    name="road_damage_2class_640",
    exist_ok=False,
)

print("Training finished.")
print("Best model:", Path(results.save_dir) / "weights" / "best.pt")
