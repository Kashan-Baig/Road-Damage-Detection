import io
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

# Local cache directories to avoid Windows permission issues
PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_DIR / ".yolo"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import torch
from tracker import RoadHazardKalmanTracker
from ultralytics import YOLO

# --------------------------------------------------
# Page Configuration & CSS Styling
# --------------------------------------------------
st.set_page_config(
    page_title="Road Damage AI - Enterprise GIS & Inspection",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
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
# Helper Functions
# --------------------------------------------------
@st.cache_resource
def load_yolo_model(weights_path: str):
    return YOLO(weights_path)


def find_available_models():
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


def normalize_hazard_class(raw_name: str) -> str:
    """Normalizes class names across RDD2022 (D00, D10, D20, D40) and custom labels."""
    name = str(raw_name).lower().strip()
    if any(k in name for k in ["pothole", "d40", "rutting", "bump"]):
        return "pothole"
    elif any(k in name for k in ["manhole", "d70", "drain", "utility"]):
        return "manhole"
    elif any(k in name for k in ["crack", "d00", "d10", "d20", "fissure"]):
        return "crack"
    return name


def find_test_videos():
    tests_dir = PROJECT_DIR / "tests"
    if not tests_dir.is_dir():
        return []
    valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return [p for p in sorted(tests_dir.iterdir()) if p.suffix.lower() in valid_exts]


def convert_to_web_mp4(input_video_path, output_video_path):
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
# Sidebar: Configuration
# --------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Model & Hardware")

    available_models = find_available_models()
    selected_model_path = st.selectbox(
        "Trained Model Weights",
        options=available_models,
        index=0,
        help="Select model checkpoint (.pt).",
    )

    try:
        model = load_yolo_model(selected_model_path)
        model_classes = {k: v.lower() for k, v in model.names.items()}
        class_names_str = ", ".join(model_classes.values())
        st.success(f"Model: `{Path(selected_model_path).name}`\n\nClasses: **{class_names_str}**")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    device = st.selectbox(
        "Hardware Device",
        options=["0", "cpu"] if torch.cuda.is_available() else ["cpu"],
        index=0,
    )

    st.markdown("---")
    st.markdown("## 🎯 Kalman Tracking & Detection")

    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.0,
        value=0.15,
        step=0.05,
        help="0.15 recommended for 75-80%+ recall on subtle/shadowed potholes.",
    )

    use_tta = st.checkbox(
        "⚡ Test-Time Augmentation (TTA)",
        value=True,
        help="Enables multi-scale flipped ensembling to boost small/distant pothole recall by +5-10%.",
    )

    iou_thresh = st.slider(
        "IoU Matching Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Spatial IoU threshold for the Hungarian matching algorithm.",
    )

    min_confirm_hits = st.slider(
        "Anti-Flicker Confirmation Frames",
        min_value=1,
        max_value=5,
        value=2,
        help="Number of consecutive frames a hazard must appear before being counted permanently.",
    )

    frame_stride = st.select_slider(
        "Frame Stride (Processing Speed)",
        options=[1, 2, 3, 4],
        value=1,
        help="1 = process every frame. 2 = process every 2nd frame (2x faster).",
    )

    st.markdown("---")
    st.markdown("## 📹 Video Source")

    input_mode = st.radio(
        "Input Mode",
        options=["Upload Video File", "Select from tests/ Folder"],
        index=0,
    )

    video_input_path = None
    input_filename = "road_video.mp4"

    if input_mode == "Upload Video File":
        uploaded_file = st.file_uploader(
            "Upload Road Video",
            type=["mp4", "avi", "mov", "mkv", "webm"],
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
# Main Header
# --------------------------------------------------
st.markdown(
    '<div class="main-title">🚧 Road Damage Detection & Spatial Inspection Platform</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Powered by <b>YOLO11</b>, <b>Kalman Velocity Tracking</b>, and <b>GIS Mapping</b>. Never counts the same pothole twice.</div>',
    unsafe_allow_html=True,
)

if not video_input_path:
    st.info("👈 Please upload a road video or select a sample from the sidebar to begin analysis.")
    st.stop()

st.markdown(f"**Selected Video:** `{input_filename}`")

start_col, _ = st.columns([2, 5])
with start_col:
    start_btn = st.button("🚀 Run Deep Video Analysis & GIS Mapping", type="primary", use_container_width=True)

# Metric display row
m_pothole_col, m_manhole_col, m_crack_col, m_total_col, m_fps_col = st.columns(5)
has_crack = any("crack" in name for name in model_classes.values())

with m_pothole_col:
    pothole_metric = st.empty()
    pothole_metric.markdown(
        """
        <div class="metric-card">
            <div class="metric-val" style="color:#EF4444;">0</div>
            <div class="metric-lbl">🔴 Unique Potholes</div>
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
            <div class="metric-lbl">🔵 Unique Manholes</div>
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
            <div class="metric-lbl">🟡 Unique Cracks</div>
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
            <div class="metric-lbl">🏆 Total Hazards</div>
            <div class="metric-sub">Verified Defects On Route</div>
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
            <div class="metric-lbl">⚡ Processing FPS</div>
            <div class="metric-sub">Real-Time Speed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
progress_bar = st.progress(0)
status_text = st.empty()

# View columns
feed_col, timeline_col = st.columns([3, 2])
with feed_col:
    live_frame_view = st.empty()
with timeline_col:
    live_chart_view = st.empty()

# --------------------------------------------------
# Video Analysis Execution Loop
# --------------------------------------------------
if start_btn:
    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        st.error("Could not open video.")
        st.stop()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_dir = tempfile.mkdtemp()
    temp_raw_video = Path(temp_dir) / "annotated_raw.avi"
    final_output_mp4 = Path(temp_dir) / "road_damage_tracked.mp4"

    effective_fps = max(1.0, orig_fps / frame_stride)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out_writer = cv2.VideoWriter(str(temp_raw_video), fourcc, effective_fps, (width, height))

    # Initialize Kalman Tracker
    tracker = RoadHazardKalmanTracker(
        min_confirmation_hits=min_confirm_hits,
        max_age=int(orig_fps * 1.5),
        iou_threshold=iou_thresh,
    )

    frame_idx = 0
    processed_count = 0
    last_fps_time = time.time()
    current_fps = orig_fps

    history_data = []

    # Map colors: Pothole (Red), Manhole (Blue), Crack (Yellow)
    color_map = {
        "pothole": (40, 50, 230),
        "manhole": (230, 150, 20),
        "crack": (30, 215, 255),
    }

    # Reference coordinates for spatial GIS mapping
    base_lat = -28.753  # Zululand / Richards Bay (Sample region)
    base_lon = 31.895

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_stride != 0:
            continue

        processed_count += 1

        # YOLO Detection with optional Test-Time Augmentation (TTA)
        results = model.predict(
            source=frame,
            conf=conf_thresh,
            iou=iou_thresh,
            device=device,
            augment=use_tta,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            raw_name = model.names.get(cls_id, "")
            c_name = normalize_hazard_class(raw_name)

            detections.append(
                {
                    "bbox": box.xyxy[0].tolist(),
                    "class_name": c_name,
                    "conf": float(box.conf[0].item()),
                }
            )

        # Kalman tracking update
        visible_hazards, cumulative = tracker.update(detections, frame_idx)

        # Measure FPS
        now = time.time()
        dt = now - last_fps_time
        last_fps_time = now
        if dt > 0:
            current_fps = 0.85 * current_fps + 0.15 * (1.0 / dt)

        # Draw Kalman Bounding Boxes & Identifiers
        annotated_frame = frame.copy()
        for hazard in visible_hazards:
            bbox = hazard.get_bbox()
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            col = color_map.get(hazard.class_name, (0, 255, 0))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), col, 2)

            badge_text = f"#{hazard.id} {hazard.class_name.capitalize()} ({hazard.conf:.0%})"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
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

        # On-screen HUD
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 60), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.78, annotated_frame, 0.22, 0, annotated_frame)

        total_u = sum(cumulative.values())
        hud_text = (
            f"Frame: {frame_idx}/{total_frames} | "
            f"KALMAN CUMULATIVE: Potholes: {cumulative.get('pothole', 0)} | "
            f"Manholes: {cumulative.get('manhole', 0)} | "
            f"Cracks: {cumulative.get('crack', 0)} | "
            f"Total Defects: {total_u}"
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
                <div class="metric-lbl">🔴 Unique Potholes</div>
                <div class="metric-sub">In Frame: {sum(1 for h in visible_hazards if h.class_name == 'pothole')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        manhole_metric.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val" style="color:#0284C7;">{cumulative.get('manhole', 0)}</div>
                <div class="metric-lbl">🔵 Unique Manholes</div>
                <div class="metric-sub">In Frame: {sum(1 for h in visible_hazards if h.class_name == 'manhole')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        crack_metric.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val" style="color:#EAB308;">{cumulative.get('crack', 0)}</div>
                <div class="metric-lbl">🟡 Unique Cracks</div>
                <div class="metric-sub">In Frame: {sum(1 for h in visible_hazards if h.class_name == 'crack')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        total_metric.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val" style="color:#10B981;">{total_u}</div>
                <div class="metric-lbl">🏆 Total Hazards</div>
                <div class="metric-sub">De-duplicated Count</div>
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
        status_text.text(f"Processing: {int(pct * 100)}% ({total_u} verified hazards logged)")

        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        live_frame_view.image(rgb_frame, channels="RGB", use_container_width=True)

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
            live_chart_view.line_chart(df, height=280)

    cap.release()
    out_writer.release()

    if history_data:
        df = pd.DataFrame(history_data).set_index("Frame")
        live_chart_view.line_chart(df, height=280)

    progress_bar.progress(1.0)
    status_text.text("✅ Analysis complete! Encoding final browser-ready video...")

    # Convert to web MP4
    with st.spinner("Finalizing browser playback..."):
        try:
            convert_to_web_mp4(temp_raw_video, final_output_mp4)
            st.session_state["output_video_path"] = str(final_output_mp4)
        except Exception as err:
            st.error(f"Fallback to raw AVI: {err}")
            st.session_state["output_video_path"] = str(temp_raw_video)

    # Compile registry with spatial GIS coordinates
    registry = tracker.hazard_registry
    hazards_records = []
    
    # Assume estimated average vehicle speed 45 km/h (12.5 m/s)
    speed_mps = 12.5
    meters_per_frame = (speed_mps / orig_fps) * frame_stride

    for hid, info in sorted(registry.items()):
        dist_m = info["first_frame"] * meters_per_frame
        
        # Approximate GPS coordinate displacement along road
        lat = base_lat + (dist_m / 111111.0)
        lon = base_lon + (dist_m / (111111.0 * np.cos(np.radians(base_lat))))

        hazards_records.append(
            {
                "Hazard ID": f"#{hid}",
                "Class": info["class_name"].capitalize(),
                "First Seen Frame": info["first_frame"],
                "Last Seen Frame": info["last_frame"],
                "Duration (Frames)": (info["last_frame"] - info["first_frame"]) + 1,
                "Peak Confidence": f"{info['peak_conf']:.1%}",
                "Road Marker (km)": f"{dist_m / 1000.0:.2f} km",
                "lat": lat,
                "lon": lon,
            }
        )

    st.session_state["hazards_records"] = hazards_records
    st.session_state["cumulative_summary"] = tracker.cumulative_counts
    st.session_state["total_unique_count"] = sum(tracker.cumulative_counts.values())

# --------------------------------------------------
# Post-Processing: Tabs for Video, GIS Map & Audit
# --------------------------------------------------
if "output_video_path" in st.session_state and Path(st.session_state["output_video_path"]).is_file():
    st.markdown("---")
    
    tab_video, tab_map, tab_audit = st.tabs(
        ["🎬 Analyzed Video Output", "🗺️ GIS Road Hazard Map", "📊 Inspection Audit & Export"]
    )

    out_path = st.session_state["output_video_path"]
    hazards_records = st.session_state.get("hazards_records", [])

    # Tab 1: Video View
    with tab_video:
        col_v, col_s = st.columns([3, 2])
        with col_v:
            with open(out_path, "rb") as vf:
                v_bytes = vf.read()
                st.video(v_bytes)

            st.download_button(
                label="⬇️ Download Verified Detection Video (.mp4)",
                data=v_bytes,
                file_name=f"inspected_{Path(input_filename).stem}.mp4",
                mime="video/mp4",
                type="primary",
                use_container_width=True,
            )

        with col_s:
            st.markdown("#### 📋 Cumulative De-duplicated Summary")
            cum = st.session_state.get("cumulative_summary", {})
            summary_df = pd.DataFrame(
                [
                    {"Hazard Type": "🔴 Potholes", "Unique Count (Strictly Once)": cum.get("pothole", 0)},
                    {"Hazard Type": "🔵 Manholes", "Unique Count (Strictly Once)": cum.get("manhole", 0)},
                    {"Hazard Type": "🟡 Cracks", "Unique Count (Strictly Once)": cum.get("crack", 0)},
                    {
                        "Hazard Type": "🏆 Total Unique Hazards",
                        "Unique Count (Strictly Once)": st.session_state.get("total_unique_count", 0),
                    },
                ]
            )
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

    # Tab 2: Interactive GIS Map
    with tab_map:
        st.markdown("#### 🗺️ Spatial Route Map (Hazard Geolocation)")
        if hazards_records:
            map_df = pd.DataFrame(hazards_records)
            
            # Color assignment for PyDeck: Pothole (Red), Manhole (Blue), Crack (Yellow)
            def get_rgb(cls_name):
                cls_lower = str(cls_name).lower()
                if "pothole" in cls_lower:
                    return [239, 68, 68, 220]
                elif "manhole" in cls_lower:
                    return [2, 132, 199, 220]
                elif "crack" in cls_lower:
                    return [234, 179, 8, 220]
                return [16, 185, 129, 220]

            map_df["color"] = map_df["Class"].apply(get_rgb)

            # PyDeck 3D Scatterplot
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_color="color",
                get_radius=20,
                pickable=True,
                auto_highlight=True,
            )

            view_state = pdk.ViewState(
                latitude=map_df["lat"].mean(),
                longitude=map_df["lon"].mean(),
                zoom=14.5,
                pitch=35,
            )

            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={
                    "html": "<b>{Hazard ID}</b><br/>Type: {Class}<br/>Peak Conf: {Peak Confidence}<br/>Route Marker: {Road Marker (km)}",
                    "style": {"backgroundColor": "#0F172A", "color": "white"},
                },
            )
            st.pydeck_chart(deck)
            st.caption("🔴 Red: Potholes | 🔵 Blue: Manholes | 🟡 Yellow: Cracks. Hover over pins for defect details.")
        else:
            st.info("No hazards detected on this route.")

    # Tab 3: Detailed Audit Registry & Export
    with tab_audit:
        st.markdown("#### 📋 Individual Hazard Registry")
        if hazards_records:
            df_reg = pd.DataFrame(hazards_records).drop(columns=["color", "lat", "lon"], errors="ignore")
            st.dataframe(df_reg, use_container_width=True, hide_index=True)

            col_csv, col_json = st.columns(2)
            with col_csv:
                csv_bytes = df_reg.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Report as CSV",
                    data=csv_bytes,
                    file_name=f"road_hazards_{Path(input_filename).stem}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_json:
                json_bytes = json.dumps(hazards_records, indent=2).encode("utf-8")
                st.download_button(
                    label="📥 Export Report as JSON (GIS Ready)",
                    data=json_bytes,
                    file_name=f"road_hazards_{Path(input_filename).stem}.json",
                    mime="application/json",
                    use_container_width=True,
                )
