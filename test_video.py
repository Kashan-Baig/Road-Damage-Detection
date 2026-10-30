
from pathlib import Path
from ultralytics import YOLO

# --------------------------------------------------
# 1. PROJECT DIRECTORY
# --------------------------------------------------
PROJECT_DIR = Path(r"C:\MyWork\RoadDamageDetection")

# --------------------------------------------------
# 2. MODEL PATH
# Change this for each model you want to test
# --------------------------------------------------
MODEL_PATH = (
    PROJECT_DIR
    / "runs"
    / "road_damage_baseline"
    / "weights"
    / "best_640.pt"
)

# --------------------------------------------------
# 3. VIDEO PATH
# Place your video in the project folder
# --------------------------------------------------
VIDEO_PATH = PROJECT_DIR / "3657637695-preview.mp4"

# --------------------------------------------------
# 4. SETTINGS
# --------------------------------------------------
CONFIDENCE = 0.25
IMAGE_SIZE = 640

# --------------------------------------------------
# 5. CHECK FILES
# --------------------------------------------------
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not VIDEO_PATH.is_file():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

# --------------------------------------------------
# 6. LOAD MODEL
# --------------------------------------------------
model = YOLO(str(MODEL_PATH))

# --------------------------------------------------
# 7. RUN VIDEO PREDICTION
# --------------------------------------------------
model.predict(
    source=str(VIDEO_PATH),
    imgsz=IMAGE_SIZE,
    conf=CONFIDENCE,
    iou=0.45,
    device="cpu",
    save=True,
    project=str(PROJECT_DIR / "runs"),
    name="video_test_conf_025",
    exist_ok=True,
    vid_stride=1
)

print("Video prediction complete.")
print("Check the output folder:")
print(PROJECT_DIR / "runs" / "video_test_conf_025")
