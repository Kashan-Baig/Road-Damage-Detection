import tempfile
import time
from pathlib import Path
import cv2
import streamlit as st
import torch
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Road Damage Detection Player",
    page_icon="🚧",
    layout="wide",
)

st.title("🚧 Road Damage Detection - Live Video Player")
st.markdown("Real-time pothole and manhole detection powered by **YOLO11**.")

# ----------------- Sidebar Controls -----------------
st.sidebar.header("⚙️ Model & Detection Settings")

default_weights = PROJECT_DIR / "best.pt"
weights_input = st.sidebar.text_input(
    "Model Weights (.pt)",
    value=str(default_weights) if default_weights.is_file() else "yolo11n.pt",
)

confidence = st.sidebar.slider(
    "Confidence Threshold", min_value=0.05, max_value=1.0, value=0.25, step=0.05
)

iou_thresh = st.sidebar.slider(
    "IoU Threshold (NMS)", min_value=0.1, max_value=0.9, value=0.45, step=0.05
)

device_choice = "0" if torch.cuda.is_available() else "cpu"
device = st.sidebar.selectbox(
    "Inference Device",
    options=["0", "cpu"] if torch.cuda.is_available() else ["cpu"],
    index=0,
)

# ----------------- Video Source Selection -----------------
st.sidebar.header("📹 Video Source")
source_mode = st.sidebar.radio(
    "Select Source", ["Test Videos in Repository", "Upload Video File"]
)

video_path = None
tests_dir = PROJECT_DIR / "tests"

if source_mode == "Test Videos in Repository":
    test_videos = (
        sorted([f for f in tests_dir.iterdir() if f.suffix.lower() in [".mp4", ".avi", ".mov"]])
        if tests_dir.is_dir()
        else []
    )
    if test_videos:
        chosen_name = st.sidebar.selectbox("Choose a test video:", [f.name for f in test_videos])
        video_path = str(tests_dir / chosen_name)
    else:
        st.sidebar.warning("No test videos found in tests/ folder.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload MP4/AVI Video", type=["mp4", "avi", "mov", "mkv"]
    )
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name

# ----------------- Main View & Detection Loop -----------------
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("📊 Live Stats")
    metric_fps = st.empty()
    metric_potholes = st.empty()
    metric_manholes = st.empty()
    metric_total = st.empty()

with col1:
    st.subheader("📺 Video Stream")
    frame_placeholder = st.empty()
    start_button = st.button("▶️ Start Detection Playback", type="primary")

if start_button:
    if not video_path:
        st.error("Please select or upload a valid video.")
        st.stop()

    if not Path(weights_input).is_file():
        st.error(f"Weights file not found at: {weights_input}")
        st.stop()

    with st.spinner("Loading YOLO model..."):
        model = YOLO(weights_input)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Failed to open video file.")
        st.stop()

    prev_time = time.time()
    fps = 0.0

    stop_placeholder = st.sidebar.button("⏹️ Stop Playback")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.info("Reached the end of video.")
            break

        # YOLO inference
        results = model.predict(
            source=frame,
            conf=confidence,
            iou=iou_thresh,
            device=device,
            verbose=False,
        )[0]

        # Draw annotations
        annotated_frame = results.plot()
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Count classes
        counts = {"pothole": 0, "manhole": 0}
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names.get(cls_id, str(cls_id)).lower()
            counts[cls_name] = counts.get(cls_name, 0) + 1

        # Calculate FPS
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Update video feed
        frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

        # Update metrics
        metric_fps.metric("Processing Speed", f"{fps:.1f} FPS")
        metric_potholes.metric("Potholes Detected", counts.get("pothole", 0))
        metric_manholes.metric("Manholes Detected", counts.get("manhole", 0))
        metric_total.metric(
            "Total Objects in Frame", counts.get("pothole", 0) + counts.get("manhole", 0)
        )

        if stop_placeholder:
            break

    cap.release()

