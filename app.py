import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

# Set cache directories to workspace to prevent permission warnings
PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_DIR / ".yolo"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import streamlit as st
import torch
from ultralytics import YOLO

# --------------------------------------------------
# Page Configuration & Styling
# --------------------------------------------------
st.set_page_config(
    page_title="Road Damage Detection AI",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Advanced Anti-Double-Count Hazard Tracker
# --------------------------------------------------
def compute_iou(boxA, boxB):
    """Compute Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    return inter / float(areaA + areaB - inter + 1e-6)


class TrackedHazard:
    """Represents a unique road hazard tracked across multiple frames."""

    def __init__(self, hazard_id, class_name, bbox, conf, frame_idx):
        self.id = hazard_id
        self.class_name = class_name
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.conf = conf
        self.peak_conf = conf
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.hits = 1  # Frames detected
        self.counted = False  # Set to True once verified, NEVER counted twice

    @property
    def centroid(self):
        return (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )


class RoadHazardTracker:
    """
    Robust multi-object tracker designed for road videos (dashcams).
    Ensures every pothole, manhole, or crack is assigned a permanent ID
    and counted strictly ONCE in the cumulative totals.
    """

    def __init__(self, min_confirmation_frames=2, max_disappeared=20):
        self.min_confirmation_frames = min_confirmation_frames
        self.max_disappeared = max_disappeared

        self.next_hazard_id = 1
        self.active_tracks = {}  # id -> TrackedHazard
        self.completed_tracks = []  # List of all unique hazards counted

        self.cumulative_counts = {"pothole": 0, "manhole": 0, "crack": 0}

    def update(self, detections, frame_idx):
        """
        detections: list of dicts {'bbox': [x1, y1, x2, y2], 'class_name': str, 'conf': float}
        """
        matched_track_ids = set()
        unmatched_detections = []

        # 1. Match current detections with active tracks using IoU & Centroid Distance
        for det in detections:
            d_bbox = det["bbox"]
            d_cls = det["class_name"]
            d_conf = det["conf"]
            d_cx = (d_bbox[0] + d_bbox[2]) / 2.0
            d_cy = (d_bbox[1] + d_bbox[3]) / 2.0

            best_tid = None
            best_iou = 0.0
            best_dist = float("inf")

            for tid, track in self.active_tracks.items():
                if tid in matched_track_ids or track.class_name != d_cls:
                    continue

                iou = compute_iou(track.bbox, d_bbox)
                tcx, tcy = track.centroid
                dist = np.hypot(d_cx - tcx, d_cy - tcy)

                # Matching criteria:
                # 1) Significant bounding box overlap (IoU >= 0.20)
                # 2) Or close spatial proximity if vehicle speed caused box shift
                box_diag = np.hypot(track.bbox[2] - track.bbox[0], track.bbox[3] - track.bbox[1])
                is_close = dist < max(60.0, box_diag * 1.5)

                if iou >= 0.20 and iou > best_iou:
                    best_iou = iou
                    best_tid = tid
                elif best_iou < 0.20 and is_close and dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                # Update matched track
                t = self.active_tracks[best_tid]
                t.bbox = d_bbox
                t.conf = d_conf
                t.peak_conf = max(t.peak_conf, d_conf)
                t.last_frame = frame_idx
                t.hits += 1
                matched_track_ids.add(best_tid)

                # Validate and count hazard strictly once
                if t.hits >= self.min_confirmation_frames and not t.counted:
                    t.counted = True
                    self.cumulative_counts[t.class_name] = (
                        self.cumulative_counts.get(t.class_name, 0) + 1
                    )
            else:
                unmatched_detections.append(det)

        # 2. Register new tracks for unmatched detections
        for det in unmatched_detections:
            new_track = TrackedHazard(
                hazard_id=self.next_hazard_id,
                class_name=det["class_name"],
                bbox=det["bbox"],
                conf=det["conf"],
                frame_idx=frame_idx,
            )
            # If min_confirmation_frames is 1, count immediately
            if self.min_confirmation_frames <= 1:
                new_track.counted = True
                self.cumulative_counts[new_track.class_name] = (
                    self.cumulative_counts.get(new_track.class_name, 0) + 1
                )

            self.active_tracks[self.next_hazard_id] = new_track
            self.next_hazard_id += 1

        # 3. Retire stale tracks that disappeared for too long
        expired_ids = []
        for tid, track in self.active_tracks.items():
            if frame_idx - track.last_frame > self.max_disappeared:
                expired_ids.append(tid)
                if track.counted:
                    self.completed_tracks.append(track)

        for tid in expired_ids:
            del self.active_tracks[tid]

        # Return live in-frame hazards and cumulative summary
        current_visible = [
            t for t in self.active_tracks.values() if t.last_frame == frame_idx
        ]
        return current_visible, self.cumulative_counts


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
@st.cache_resource
def load_yolo_model(weights_path: str):
    """Loads and caches the YOLO model in memory."""
    return YOLO(weights_path)


def find_available_models():
    """Finds .pt model files in the project."""
    models = []
    candidates = [
        PROJECT_DIR / "best.pt",
        PROJECT_DIR / "runs" / "road_damage_baseline" / "weights" / "best_640.pt",
        PROJECT_DIR / "runs" / "road_damage_2class_640" / "weights" / "best.pt",
        PROJECT_DIR / "yolo11n.pt",
    ]
    for c in candidates:
        if c.is_file() and str(c) not in models:
            models.append(str(c))

    for extra in PROJECT_DIR.glob("*.pt"):
        s = str(extra)
        if s not in models:
            models.append(s)

    return models if models else ["best.pt"]


def find_test_videos():
    """Finds sample test videos inside tests/ directory."""
    tests_dir = PROJECT_DIR / "tests"
    if not tests_dir.is_dir():
        return []
    valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return [p for p in sorted(tests_dir.iterdir()) if p.suffix.lower() in valid_exts]


def convert_to_web_mp4(input_video_path, output_video_path):
    """Converts a video to H.264 MP4 using imageio-ffmpeg for browser compatibility."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(input_video_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        str(output_video_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


# --------------------------------------------------
# Sidebar: Settings & Video Source
# --------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Model Configuration")

    available_models = find_available_models()
    selected_model_path = st.selectbox(
        "Trained Model Weights",
        options=available_models,
        index=0,
        help="Select the trained YOLO weights checkpoint.",
    )

    try:
        model = load_yolo_model(selected_model_path)
        model_classes = {k: v.lower() for k, v in model.names.items()}
        class_names_str = ", ".join(model_classes.values())
        st.success(f"Model: `{Path(selected_model_path).name}`\n\nClasses: **{class_names_str}**")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    st.markdown("---")
    st.markdown("## 🎯 Tracking & Detection Parameters")

    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.0,
        value=0.20,
        step=0.05,
        help="Confidence threshold for YOLO bounding box proposals.",
    )

    iou_thresh = st.slider(
        "NMS IoU Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
    )

    min_confirm_frames = st.slider(
        "Confirmation Frames (Anti-Flicker)",
        min_value=1,
        max_value=5,
        value=2,
        help="A hazard must appear in this many consecutive frames before being permanently counted (prevents false-positive flicker).",
    )

    frame_stride = st.select_slider(
        "Frame Stride (Processing Speed)",
        options=[1, 2, 3, 4],
        value=1,
        help="Stride 1 processes all frames. Stride 2 processes every 2nd frame (2x faster).",
    )

    device = st.selectbox(
        "Hardware Acceleration",
        options=["0", "cpu"] if torch.cuda.is_available() else ["cpu"],
        index=0,
    )

    st.markdown("---")
    st.markdown("## 📁 Video Selection")

    input_mode = st.radio(
        "Input Source",
        options=["Upload Video File", "Select from tests/ Folder"],
        index=0,
    )

    video_input_path = None
    input_filename = "video.mp4"

    if input_mode == "Upload Video File":
        uploaded_file = st.file_uploader(
            "Upload Road Video",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Upload dashcam or road footage for hazard detection.",
        )
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded_file.name).suffix
            )
            tfile.write(uploaded_file.read())
            video_input_path = tfile.name
            input_filename = uploaded_file.name
    else:
        sample_videos = find_test_videos()
        if sample_videos:
            chosen_video = st.selectbox(
                "Choose Sample Video",
                options=sample_videos,
                format_func=lambda p: p.name,
            )
            if chosen_video:
                video_input_path = str(chosen_video)
                input_filename = chosen_video.name
        else:
            st.warning("No sample videos found in `tests/` folder.")


# --------------------------------------------------
# Main Dashboard Header
# --------------------------------------------------
st.markdown(
    '<div class="main-title">🚧 Road Damage Detection & Anti-Duplicate Hazard Counter</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Multi-object tracking ensures each <b>Pothole</b>, <b>Manhole</b>, and <b>Crack</b> receives a unique ID and is counted <b>strictly once</b> across the entire video.</div>',
    unsafe_allow_html=True,
)

if not video_input_path:
    st.info("👈 Please upload a road video or select a sample from the sidebar to begin.")
    st.stop()

# --------------------------------------------------
# Live Cumulative Metric Cards
# --------------------------------------------------
st.markdown(f"**Selected Video:** `{input_filename}`")

start_col, _ = st.columns([2, 5])
with start_col:
    start_btn = st.button("🚀 Start Video Analysis", type="primary", use_container_width=True)

m_pothole_col, m_manhole_col, m_crack_col, m_total_col, m_fps_col = st.columns(5)

has_crack = any("crack" in name for name in model_classes.values())

with m_pothole_col:
    pothole_metric = st.empty()
    pothole_metric.markdown(
        """
        <div class="metric-card">
            <div class="metric-val" style="color:#EF4444;">0</div>
            <div class="metric-lbl">🔴 Cumulative Potholes</div>
            <div class="metric-sub">Counted Once (No Duplicates)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_manhole_col:
    manhole_metric = st.empty()
    manhole_metric.markdown(
        """
        <div class="metric-card">
            <div class="metric-val" style="color:#0284C7;">0</div>
            <div class="metric-lbl">🔵 Cumulative Manholes</div>
            <div class="metric-sub">Counted Once (No Duplicates)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_crack_col:
    crack_metric = st.empty()
    crack_sub = "Counted Once" if has_crack else "N/A (2-Class Model)"
    crack_metric.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-val" style="color:#EAB308;">0</div>
            <div class="metric-lbl">🟡 Cumulative Cracks</div>
            <div class="metric-sub">{crack_sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_total_col:
    total_metric = st.empty()
    total_metric.markdown(
        """
        <div class="metric-card">
            <div class="metric-val" style="color:#10B981;">0</div>
            <div class="metric-lbl">🏆 Total Unique Hazards</div>
            <div class="metric-sub">Total Individual Road Defects</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_fps_col:
    fps_metric = st.empty()
    fps_metric.markdown(
        """
        <div class="metric-card">
            <div class="metric-val" style="color:#6366F1;">0.0</div>
            <div class="metric-lbl">⚡ Processing Speed</div>
            <div class="metric-sub">Real-Time FPS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
progress_bar = st.progress(0)
status_text = st.empty()

video_feed_col, chart_col = st.columns([3, 2])
with video_feed_col:
    live_frame_view = st.empty()
with chart_col:
    chart_view = st.empty()

# --------------------------------------------------
# Video Analysis Execution Loop
# --------------------------------------------------
if start_btn:
    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        st.error("Could not open video file.")
        st.stop()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_dir = tempfile.mkdtemp()
    temp_raw_video = Path(temp_dir) / "raw_annotated.avi"
    final_output_mp4 = Path(temp_dir) / "road_damage_detected.mp4"

    effective_fps = max(1.0, orig_fps / frame_stride)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out_writer = cv2.VideoWriter(str(temp_raw_video), fourcc, effective_fps, (width, height))

    tracker = RoadHazardTracker(
        min_confirmation_frames=min_confirm_frames,
        max_disappeared=int(orig_fps * 1.5),  # Remember hazard for 1.5 seconds if briefly lost
    )

    frame_idx = 0
    processed_count = 0
    last_fps_time = time.time()
    current_fps = orig_fps

    history_data = []

    # Color palette (BGR): Pothole=Red/Coral, Manhole=Bright Blue, Crack=Yellow
    color_map = {
        "pothole": (40, 50, 230),   # Red / Coral
        "manhole": (230, 150, 20),  # Blue / Cyan
        "crack": (30, 215, 255),    # Yellow
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_stride != 0:
            continue

        processed_count += 1

        # YOLO Inference
        results = model.predict(
            source=frame,
            conf=conf_thresh,
            iou=iou_thresh,
            device=device,
            verbose=False,
        )[0]

        # Extract detections for the tracker
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            raw_name = model.names.get(cls_id, "").lower()

            if "pothole" in raw_name:
                c_name = "pothole"
            elif "manhole" in raw_name:
                c_name = "manhole"
            elif "crack" in raw_name:
                c_name = "crack"
            else:
                c_name = raw_name

            detections.append(
                {
                    "bbox": box.xyxy[0].tolist(),
                    "class_name": c_name,
                    "conf": float(box.conf[0].item()),
                }
            )

        # Update the tracker
        visible_hazards, cumulative = tracker.update(detections, frame_idx)

        # Calculate FPS
        now = time.time()
        dt = now - last_fps_time
        last_fps_time = now
        if dt > 0:
            current_fps = 0.85 * current_fps + 0.15 * (1.0 / dt)

        # Render custom bounding boxes with persistent IDs
        annotated_frame = frame.copy()
        for hazard in visible_hazards:
            x1, y1, x2, y2 = [int(v) for v in hazard.bbox]
            col = color_map.get(hazard.class_name, (0, 255, 0))

            # Bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), col, 2)

            # Label badge: e.g. "#4 Pothole (85%)"
            badge_text = f"#{hazard.id} {hazard.class_name.capitalize()} {hazard.conf:.0%}"
            (tw, th), baseline = cv2.getTextSize(
                badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            cv2.rectangle(
                annotated_frame,
                (x1, max(0, y1 - th - 8)),
                (x1 + tw + 6, y1),
                col,
                -1,
            )
            cv2.putText(
                annotated_frame,
                badge_text,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # On-screen persistent HUD banner
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 60), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.78, annotated_frame, 0.22, 0, annotated_frame)

        total_unique_count = sum(cumulative.values())
        hud_text = (
            f"Frame: {frame_idx}/{total_frames} | "
            f"CUMULATIVE UNIQUE: Potholes: {cumulative.get('pothole', 0)} | "
            f"Manholes: {cumulative.get('manhole', 0)} | "
            f"Cracks: {cumulative.get('crack', 0)} | "
            f"Total Hazards: {total_unique_count}"
        )
        cv2.putText(
            annotated_frame,
            hud_text,
            (15, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_writer.write(annotated_frame)

        # Update metric cards
        pothole_metric.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val" style="color:#EF4444;">{cumulative.get('pothole', 0)}</div>
                <div class="metric-lbl">🔴 Cumulative Potholes</div>
                <div class="metric-sub">In Frame: {sum(1 for h in visible_hazards if h.class_name == 'pothole')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        manhole_metric.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val" style="color:#0284C7;">{cumulative.get('manhole', 0)}</div>
                <div class="metric-lbl">🔵 Cumulative Manholes</div>
                <div class="metric-sub">In Frame: {sum(1 for h in visible_hazards if h.class_name == 'manhole')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        crack_metric.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val" style="color:#EAB308;">{cumulative.get('crack', 0)}</div>
                <div class="metric-lbl">🟡 Cumulative Cracks</div>
                <div class="metric-sub">In Frame: {sum(1 for h in visible_hazards if h.class_name == 'crack')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        total_metric.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val" style="color:#10B981;">{total_unique_count}</div>
                <div class="metric-lbl">🏆 Total Unique Hazards</div>
                <div class="metric-sub">Unique Count Across Video</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fps_metric.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val" style="color:#6366F1;">{current_fps:.1f}</div>
                <div class="metric-lbl">⚡ Processing Speed</div>
                <div class="metric-sub">Frame: {frame_idx}/{total_frames}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pct = min(1.0, frame_idx / total_frames)
        progress_bar.progress(pct)
        status_text.text(
            f"Processing video: {int(pct * 100)}% (Logged {total_unique_count} unique hazards without duplicates)"
        )

        # Show live visual feed
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        live_frame_view.image(rgb_frame, channels="RGB", use_container_width=True)

        # History for cumulative progression chart
        history_data.append(
            {
                "Frame": frame_idx,
                "Cumulative Potholes": cumulative.get("pothole", 0),
                "Cumulative Manholes": cumulative.get("manhole", 0),
                "Cumulative Cracks": cumulative.get("crack", 0),
            }
        )

        if processed_count % 15 == 0:
            df = pd.DataFrame(history_data).set_index("Frame")
            chart_view.line_chart(df, height=280)

    cap.release()
    out_writer.release()

    if history_data:
        df = pd.DataFrame(history_data).set_index("Frame")
        chart_view.line_chart(df, height=280)

    progress_bar.progress(1.0)
    status_text.text("✅ Analysis complete! Encoding browser video output...")

    # Encode to web-friendly MP4
    with st.spinner("Finalizing browser-compatible video..."):
        try:
            convert_to_web_mp4(temp_raw_video, final_output_mp4)
            st.session_state["output_video_path"] = str(final_output_mp4)
        except Exception as err:
            st.error(f"Video conversion fallback: {err}")
            st.session_state["output_video_path"] = str(temp_raw_video)

    # Compile the complete list of unique hazards
    all_unique_hazards = tracker.completed_tracks + [
        t for t in tracker.active_tracks.values() if t.counted
    ]
    all_unique_hazards.sort(key=lambda x: x.id)

    st.session_state["all_hazards"] = all_unique_hazards
    st.session_state["cumulative_summary"] = tracker.cumulative_counts
    st.session_state["total_unique_count"] = sum(tracker.cumulative_counts.values())

# --------------------------------------------------
# Video Output & Detailed De-duplicated Log
# --------------------------------------------------
if "output_video_path" in st.session_state and Path(st.session_state["output_video_path"]).is_file():
    st.markdown("---")
    st.markdown("### 🎬 Analyzed Video with Unique Hazard Tracking")

    out_path = st.session_state["output_video_path"]
    col_out, col_summary = st.columns([3, 2])

    with col_out:
        with open(out_path, "rb") as vf:
            video_bytes = vf.read()
            st.video(video_bytes)

        st.download_button(
            label="⬇️ Download Annotated Video with Unique IDs (.mp4)",
            data=video_bytes,
            file_name=f"tracked_{Path(input_filename).stem}.mp4",
            mime="video/mp4",
            type="primary",
            use_container_width=True,
        )

    with col_summary:
        st.markdown("#### 📋 Cumulative De-duplicated Summary")
        cum = st.session_state.get("cumulative_summary", {})

        summary_df = pd.DataFrame(
            [
                {"Hazard Type": "🔴 Potholes", "Unique Count (De-duplicated)": cum.get("pothole", 0)},
                {"Hazard Type": "🔵 Manholes", "Unique Count (De-duplicated)": cum.get("manhole", 0)},
                {"Hazard Type": "🟡 Cracks", "Unique Count (De-duplicated)": cum.get("crack", 0)},
                {
                    "Hazard Type": "🏆 Total Unique Hazards",
                    "Unique Count (De-duplicated)": st.session_state.get("total_unique_count", 0),
                },
            ]
        )
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

    # Detailed table of every unique hazard encountered
    hazards_list = st.session_state.get("all_hazards", [])
    if hazards_list:
        st.markdown("#### 🔎 Individual Hazard Registry (Proof of No Double-Counting)")
        records = []
        for h in hazards_list:
            records.append(
                {
                    "Hazard ID": f"#{h.id}",
                    "Class": h.class_name.capitalize(),
                    "First Seen Frame": h.first_frame,
                    "Last Seen Frame": h.last_frame,
                    "Duration (Frames)": (h.last_frame - h.first_frame) + 1,
                    "Peak Confidence": f"{h.peak_conf:.1%}",
                }
            )
        registry_df = pd.DataFrame(records)
        st.dataframe(registry_df, use_container_width=True, hide_index=True)
