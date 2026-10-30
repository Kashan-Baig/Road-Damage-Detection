
from pathlib import Path
from ultralytics import YOLO

# --------------------------------------------------
# 1. PROJECT DIRECTORY
# --------------------------------------------------
PROJECT_DIR = Path(r"C:\MyWork\RoadDamageDetection")

# --------------------------------------------------
# 2. SELECT MODEL TO TEST
# Change the model folder/name for each experiment
# --------------------------------------------------
MODEL_PATH = (
    PROJECT_DIR
    / "runs"
    / "road_damage_baseline"
    / "weights"
    / "best_640.pt"
)

# --------------------------------------------------
# 3. SELECT TEST IMAGE
# --------------------------------------------------
IMAGE_PATH = (
    PROJECT_DIR
    / "road_damage_2class"
    / "images"
    / "test"
    / "vlcsnap-2025-02-19-15h05m20s943.jpg"
)

# --------------------------------------------------
# 4. CHECK PATHS
# --------------------------------------------------
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not IMAGE_PATH.is_file():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# --------------------------------------------------
# 5. LOAD MODEL
# --------------------------------------------------
model = YOLO(str(MODEL_PATH))

# --------------------------------------------------
# 6. RUN PREDICTION
# --------------------------------------------------
CONFIDENCE = 0.25  # Try 0.10, 0.15, 0.25, 0.40, 0.50

model.predict(
    source=str(IMAGE_PATH),
    imgsz=640,
    conf=CONFIDENCE,
    iou=0.45,
    device="cpu",
    save=True,
    project=str(PROJECT_DIR / "runs"),
    name=f"predictions_conf_{str(CONFIDENCE).replace('.', '')}",
    exist_ok=True,
)

print("Prediction complete.")
print("Results saved inside:", PROJECT_DIR / "runs")
