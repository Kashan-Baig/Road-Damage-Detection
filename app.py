import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

# Local cache directories to avoid Windows permission issues
PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_DIR / ".yolo"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))

import altair as alt
import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import torch
from src.tracker import RoadHazardKalmanTracker
from ultralytics import YOLO

# ==================================================
# Page configuration
# ==================================================
st.set_page_config(
    page_title="RoadSense AI · Hazard Inspection",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# Theme system
# ==================================================
THEMES = {
    "dark": dict(
        bg="#0A0E13", bg2="#0E131A", surface="#121922", surface2="#19222E", border="#243041",
        text="#E7EDF4", muted="#8B9AAE", faint="#5B6A7E", accent="#F5B301", accent_ink="#111111",
        pothole="#FF5A5F", manhole="#38BDF8", crack="#FACC15", ok="#22C55E", violet="#A78BFA",
        grid="#1D2734", shadow="0 10px 30px rgba(0,0,0,.35)", map="dark",
    ),
    "light": dict(
        bg="#F5F2EB", bg2="#EFEBE2", surface="#FFFFFF", surface2="#F8F6F1", border="#E3DDD1",
        text="#15181D", muted="#5E6876", faint="#9AA3AF", accent="#D98E04", accent_ink="#111111",
        pothole="#DC2626", manhole="#0284C7", crack="#CA8A04", ok="#16A34A", violet="#7C3AED",
        grid="#ECE7DC", shadow="0 8px 24px rgba(30,25,10,.08)", map="light",
    ),
}

# BGR colours used when drawing on video frames (theme-independent, high contrast)
DRAW_BGR = {
    "pothole": (95, 90, 255),
    "manhole": (248, 189, 56),
    "crack": (21, 204, 250),
}
CLASS_ICON = {"pothole": "🔴", "manhole": "🔵", "crack": "🟡"}
PRIORITY_ORDER = ["High", "Medium", "Low"]

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, .stApp { color: var(--text); font-family: 'Inter', system-ui, sans-serif; }
.stApp {
  background:
    radial-gradient(1100px 520px at 88% -12%, color-mix(in srgb, var(--accent) 8%, transparent), transparent 60%),
    radial-gradient(900px 480px at -10% 110%, color-mix(in srgb, var(--manhole) 6%, transparent), transparent 60%),
    var(--bg) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1520px; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 { color: var(--text); font-family: 'Space Grotesk', sans-serif; }
.stApp p, .stApp li, .stApp label, [data-testid="stWidgetLabel"] p, .stMarkdown { color: var(--text); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
hr { border-color: var(--border) !important; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: var(--muted); font-size: .8rem; font-weight: 600; }
[data-testid="stSidebar"] .stMarkdown p { color: var(--text); }
.side-sec { display:flex; align-items:center; gap:8px; margin: 18px 0 6px; font-size:.7rem; font-weight:700;
  letter-spacing:.14em; text-transform:uppercase; color: var(--faint); }
.side-sec::after { content:""; flex:1; height:1px; background: var(--border); }
.brand { display:flex; align-items:center; gap:12px; padding: 4px 0 10px; }
.brand-logo { width:40px; height:40px; border-radius:12px; display:grid; place-items:center; font-size:1.25rem;
  background: repeating-linear-gradient(-45deg, var(--accent) 0 7px, #111 7px 14px); box-shadow: var(--shadow); }
.brand-name { font-family:'Space Grotesk'; font-weight:700; font-size:1.15rem; color:var(--text); line-height:1.1; }
.brand-tag { font-size:.72rem; color:var(--muted); }
.model-card { background: var(--surface); border:1px solid var(--border); border-radius:12px; padding:10px 12px; margin-top:6px; font-size:.8rem; }
.model-card b { color: var(--text); }
.model-card .mc-row { display:flex; justify-content:space-between; color: var(--muted); margin-top:3px; }

/* ---------- Inputs ---------- */
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="base-input"],
.stNumberInput input, .stTextInput input {
  background: var(--surface2) !important; border-color: var(--border) !important; color: var(--text) !important; }
[data-baseweb="select"] span, [data-baseweb="select"] div { color: var(--text) !important; }
[data-baseweb="select"] svg { fill: var(--muted) !important; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"] { background: var(--surface2) !important; color: var(--text) !important; }
[data-baseweb="popover"] li:hover { background: var(--surface) !important; }
[data-baseweb="tag"] { background: color-mix(in srgb, var(--accent) 22%, transparent) !important; }
[data-baseweb="tag"] span { color: var(--text) !important; }
.stNumberInput button { background: var(--surface2) !important; color: var(--text) !important; border-color: var(--border) !important; }
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: var(--muted) !important; font-family:'JetBrains Mono'; }
[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; border-color: var(--accent) !important; box-shadow: 0 0 0 6px color-mix(in srgb, var(--accent) 20%, transparent) !important; }
[data-testid="stFileUploaderDropzone"] { background: var(--surface) !important; border: 1px dashed var(--border) !important; border-radius: 12px; }
[data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] button { background: var(--surface2) !important; color: var(--text) !important; border: 1px solid var(--border) !important; }
[data-testid="stFileUploaderFile"] * { color: var(--text) !important; }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
  border-radius: 12px; border: 1px solid var(--border); background: var(--surface); color: var(--text);
  font-weight: 600; padding: .6rem 1rem; transition: all .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--accent); color: var(--text); transform: translateY(-1px); }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--accent), color-mix(in srgb, var(--accent) 70%, var(--pothole))) !important;
  color: var(--accent_ink) !important; border: none !important; box-shadow: 0 8px 24px color-mix(in srgb, var(--accent) 35%, transparent); }
.stButton > button[kind="primary"] p, .stDownloadButton > button[kind="primary"] p { color: var(--accent_ink) !important; font-weight: 700; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 10px 10px 0 0; padding: 10px 18px; }
.stTabs [data-baseweb="tab"] p { color: var(--muted); font-weight: 600; }
.stTabs [aria-selected="true"] { background: var(--surface) !important; }
.stTabs [aria-selected="true"] p { color: var(--text) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent) !important; height: 3px; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- Misc widgets ---------- */
[data-testid="stExpander"] details { background: var(--surface); border: 1px solid var(--border) !important; border-radius: 12px; }
[data-testid="stExpander"] summary p { color: var(--text); font-weight: 600; }
[data-testid="stExpander"] summary svg { fill: var(--muted); }
[data-testid="stAlert"] { background: var(--surface) !important; border: 1px solid var(--border); border-radius: 12px; }
[data-testid="stAlert"] p { color: var(--text) !important; }
[data-testid="stSpinner"] p, [data-testid="stSpinner"] div { color: var(--text) !important; }
[data-testid="stProgress"] [role="progressbar"] > div { background: var(--surface2) !important; }
[data-testid="stProgress"] [role="progressbar"] > div > div { background: linear-gradient(90deg, var(--accent), var(--pothole)) !important; }
[data-testid="stImage"] img, video { border-radius: 14px; }
[data-testid="stVideo"] { border-radius: 14px; overflow: hidden; border: 1px solid var(--border); }
[data-testid="stDeckGlJsonChart"] { border-radius: 16px; overflow: hidden; border: 1px solid var(--border); }
[data-testid="stToggle"] p, [data-testid="stCheckbox"] p, [data-testid="stRadio"] p { color: var(--text) !important; }

/* ---------- Hero ---------- */
.hero { position: relative; background: var(--surface); border: 1px solid var(--border); border-radius: 20px;
  padding: 22px 26px 20px; box-shadow: var(--shadow); overflow: hidden; margin-bottom: 18px; }
.hero::before { content:""; position:absolute; inset:0 0 auto 0; height:5px;
  background: repeating-linear-gradient(-45deg, var(--accent) 0 14px, #111 14px 28px); }
.hero-row { display:flex; justify-content:space-between; align-items:flex-end; gap:20px; flex-wrap:wrap; }
.eyebrow { font-size:.72rem; font-weight:700; letter-spacing:.18em; text-transform:uppercase; color: var(--accent); }
.hero h1 { font-family:'Space Grotesk'; font-size: 2.05rem; font-weight: 700; margin: 4px 0 4px; color: var(--text); letter-spacing:-.02em; padding:0; }
.hero p.lead { color: var(--muted); margin: 0; font-size: .98rem; max-width: 760px; }
.chips { display:flex; gap:8px; flex-wrap:wrap; }
.chip { display:inline-flex; align-items:center; gap:6px; padding:5px 11px; border-radius:999px; font-size:.76rem; font-weight:600;
  background: var(--surface2); border:1px solid var(--border); color: var(--text); white-space:nowrap; }
.chip .dot { width:7px; height:7px; border-radius:50%; background: var(--c, var(--ok)); }

/* ---------- KPI cards ---------- */
.kpi { position:relative; background: var(--surface); border:1px solid var(--border); border-radius:16px;
  padding:16px 18px 14px; box-shadow: var(--shadow); overflow:hidden; min-height: 118px; }
.kpi::before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background: var(--kc); }
.kpi::after { content:""; position:absolute; right:-36px; top:-36px; width:120px; height:120px; border-radius:50%;
  background: radial-gradient(circle, color-mix(in srgb, var(--kc) 24%, transparent), transparent 70%); }
.kpi-top { display:flex; align-items:center; gap:8px; color: var(--muted); font-size:.72rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
.kpi-dot { width:8px; height:8px; border-radius:50%; background: var(--kc); box-shadow: 0 0 0 4px color-mix(in srgb, var(--kc) 20%, transparent); }
.kpi-val { font-family:'JetBrains Mono', monospace; font-size:2.1rem; font-weight:700; color: var(--text); margin-top:8px; line-height:1.05; }
.kpi-unit { font-size:.85rem; color: var(--muted); margin-left:4px; font-family:'Inter'; font-weight:600; }
.kpi-sub { font-size:.78rem; color: var(--muted); margin-top:6px; }
.kpi-sub b { color: var(--kc); }

/* ---------- Panels & status ---------- */
.panel-title { display:flex; align-items:center; justify-content:space-between; margin: 6px 0 10px; }
.panel-title h3 { font-family:'Space Grotesk'; font-size:1.05rem; font-weight:700; margin:0; padding:0; color: var(--text); }
.panel-title span { font-size:.75rem; color: var(--muted); }
.live { display:inline-flex; align-items:center; gap:6px; font-size:.72rem; font-weight:700; letter-spacing:.1em; color: var(--pothole); }
.live::before { content:""; width:8px; height:8px; border-radius:50%; background: var(--pothole); animation: pulse 1.2s infinite; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--pothole) 60%, transparent); } 100% { box-shadow: 0 0 0 10px transparent; } }
.statusbar { display:flex; gap:18px; flex-wrap:wrap; align-items:center; background: var(--surface); border:1px solid var(--border);
  border-radius:12px; padding:10px 14px; font-size:.82rem; color: var(--muted); }
.statusbar b { color: var(--text); font-family:'JetBrains Mono'; }
.placeholder { display:grid; place-items:center; text-align:center; height: 360px; border:1px dashed var(--border);
  border-radius:16px; background: var(--surface); color: var(--muted); font-size:.9rem; }
.placeholder .big { font-size: 2.2rem; margin-bottom: 6px; }
.src-bar { display:flex; align-items:center; gap:14px; background: var(--surface); border:1px solid var(--border); border-radius:14px; padding:12px 16px; }
.src-ico { width:42px; height:42px; border-radius:10px; display:grid; place-items:center; background: var(--surface2); font-size:1.2rem; }
.src-name { font-weight:700; color: var(--text); font-size:.95rem; word-break: break-all; }
.src-meta { color: var(--muted); font-size:.78rem; font-family:'JetBrains Mono'; }

/* ---------- Empty state ---------- */
.steps { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:14px; margin-top: 6px; }
.step { background: var(--surface); border:1px solid var(--border); border-radius:16px; padding:18px; box-shadow: var(--shadow); }
.step .n { font-family:'JetBrains Mono'; font-weight:700; color: var(--accent); font-size:.85rem; }
.step h4 { margin:6px 0 4px; font-size:1.05rem; padding:0; }
.step p { margin:0; color: var(--muted) !important; font-size:.86rem; }
@media (max-width: 900px) { .steps { grid-template-columns: 1fr; } }

/* ---------- Stat tiles ---------- */
.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:10px; }
.tile { background: var(--surface); border:1px solid var(--border); border-radius:14px; padding:12px 14px; }
.tile .l { font-size:.7rem; color: var(--muted); font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
.tile .v { font-family:'JetBrains Mono'; font-size:1.3rem; font-weight:700; color: var(--text); margin-top:4px; }

/* ---------- Registry table ---------- */
.reg-wrap { max-height: 560px; overflow:auto; border:1px solid var(--border); border-radius:14px; background: var(--surface); }
.reg { width:100%; border-collapse: collapse; font-size:.85rem; }
.reg th { position: sticky; top:0; background: var(--surface2); color: var(--muted); text-align:left; font-size:.7rem;
  letter-spacing:.1em; text-transform:uppercase; padding:11px 12px; border-bottom:1px solid var(--border); z-index:1; }
.reg td { padding:10px 12px; border-top:1px solid var(--border); color: var(--text); white-space:nowrap; }
.reg tr:hover td { background: color-mix(in srgb, var(--accent) 6%, transparent); }
.mono { font-family:'JetBrains Mono'; }
.pill { display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:999px; font-weight:700; font-size:.74rem;
  background: color-mix(in srgb, var(--c) 15%, transparent); color: var(--c); border: 1px solid color-mix(in srgb, var(--c) 35%, transparent); }
.bar { display:inline-block; vertical-align:middle; height:6px; width:84px; background: var(--surface2); border-radius:4px; margin-right:8px; overflow:hidden; }
.bar > span { display:block; height:100%; background: var(--c); border-radius:4px; }

/* ---------- Legend & evidence ---------- */
.legend { display:flex; flex-direction:column; gap:10px; background: var(--surface); border:1px solid var(--border); border-radius:14px; padding:14px; }
.legend-row { display:flex; justify-content:space-between; align-items:center; font-size:.86rem; color: var(--text); }
.legend-row .sw { display:inline-block; width:12px; height:12px; border-radius:50%; background: var(--c); margin-right:8px; vertical-align:middle; }
.ev-cap { display:flex; justify-content:space-between; align-items:center; margin: 4px 2px 16px; font-size:.78rem; color: var(--muted); }
.ev-cap b { color: var(--text); font-family:'JetBrains Mono'; }
"""


def inject_theme(T: dict):
    vars_css = ";".join(f"--{k}:{v}" for k, v in T.items() if not k in ("shadow", "map")) + f";--shadow:{T['shadow']}"
    st.markdown(f"<style>:root{{{vars_css}}}{BASE_CSS}</style>", unsafe_allow_html=True)


def html(s: str):
    st.markdown(s, unsafe_allow_html=True)


def class_color(cls: str, T: dict) -> str:
    return T.get(str(cls).lower(), T["ok"])


def priority_color(p: str, T: dict) -> str:
    return {"High": T["pothole"], "Medium": T["accent"], "Low": T["ok"]}.get(p, T["muted"])


def hex_to_rgba(h: str, a: int = 230):
    h = h.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)] + [a]


def style_chart(chart, T: dict):
    return (
        chart.configure(background="transparent", font="Inter")
        .configure_axis(
            labelColor=T["muted"], titleColor=T["muted"], gridColor=T["grid"],
            domainColor=T["border"], tickColor=T["border"], labelFontSize=11, titleFontSize=11,
        )
        .configure_legend(labelColor=T["text"], titleColor=T["muted"], orient="top", labelFontSize=11)
        .configure_view(strokeWidth=0)
    )


# ==================================================
# Helpers: model, video, geometry
# ==================================================
@st.cache_resource(show_spinner=False)
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
        if str(extra) not in models:
            models.append(str(extra))
    return models if models else ["best.pt"]


def normalize_hazard_class(raw_name: str) -> str:
    """Normalizes class names across RDD2022 (D00, D10, D20, D40) and custom labels."""
    name = str(raw_name).lower().strip()
    if any(k in name for k in ["pothole", "d40", "rutting", "bump"]):
        return "pothole"
    if any(k in name for k in ["manhole", "d70", "drain", "utility"]):
        return "manhole"
    if any(k in name for k in ["crack", "d00", "d10", "d20", "fissure"]):
        return "crack"
    return name


def find_test_videos():
    tests_dir = PROJECT_DIR / "tests"
    if not tests_dir.is_dir():
        return []
    valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return [p for p in sorted(tests_dir.iterdir()) if p.suffix.lower() in valid_exts]


@st.cache_data(show_spinner=False)
def probe_video(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": n, "w": w, "h": h, "duration": n / fps if fps else 0}


def convert_to_web_mp4(input_video_path, output_video_path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y", "-i", str(input_video_path),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output_video_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def offset_coords(lat0, lon0, dist_m, bearing_deg):
    b = np.radians(bearing_deg)
    dlat = dist_m * np.cos(b) / 111111.0
    dlon = dist_m * np.sin(b) / (111111.0 * np.cos(np.radians(lat0)))
    return lat0 + dlat, lon0 + dlon


def priority_of(cls: str, conf: float, area_ratio: float):
    """Repair priority from detector confidence and apparent size in frame."""
    size = min(1.0, area_ratio / 0.02)  # a box covering ~2% of the frame counts as 'large'
    score = 0.55 * conf + 0.45 * size
    if cls == "manhole":
        score *= 0.8  # utility asset rather than surface failure
    label = "High" if score >= 0.6 else "Medium" if score >= 0.38 else "Low"
    return score, label


def fmt_dur(sec: float) -> str:
    sec = max(0, int(sec))
    m, s = divmod(sec, 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def draw_hazard(img, box, color, text):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    L = max(8, int(min(x2 - x1, y2 - y1) * 0.25))
    for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, 3, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, 3, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = y1 - 8 if y1 - th - 10 > 0 else y2 + th + 10
    cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 10, ty + 4), color, -1)
    cv2.putText(img, text, (x1 + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 1, cv2.LINE_AA)


def draw_hud(img, width, frame_idx, total_frames, cum):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (width, 46), (19, 14, 10), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.rectangle(img, (0, 46), (int(width * frame_idx / max(1, total_frames)), 49), (1, 179, 245), -1)
    left = f"POTHOLES {cum.get('pothole', 0)}   MANHOLES {cum.get('manhole', 0)}   TOTAL {sum(cum.values())}"
    cv2.putText(img, left, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    right = f"FRAME {frame_idx}/{total_frames}"
    (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(img, right, (width - tw - 16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 200, 210), 1, cv2.LINE_AA)


# ==================================================
# UI fragments
# ==================================================
def kpi_html(label, value, sub, color_var, unit=""):
    unit_html = f'<span class="kpi-unit">{unit}</span>' if unit else ""
    return (
        f'<div class="kpi" style="--kc:var({color_var})">'
        f'<div class="kpi-top"><span class="kpi-dot"></span>{label}</div>'
        f'<div class="kpi-val">{value}{unit_html}</div>'
        f'<div class="kpi-sub">{sub}</div></div>'
    )


def render_kpis(slots, cum, in_frame, density, fps_val, fps_sub):
    p, m = cum.get("pothole", 0), cum.get("manhole", 0)
    total = sum(cum.values())
    slots[0].markdown(kpi_html("Unique Potholes", p, f"In frame now: <b>{in_frame.get('pothole', 0)}</b>", "--pothole"), unsafe_allow_html=True)
    slots[1].markdown(kpi_html("Unique Manholes", m, f"In frame now: <b>{in_frame.get('manhole', 0)}</b>", "--manhole"), unsafe_allow_html=True)
    slots[2].markdown(kpi_html("Total Hazards", total, "Kalman de-duplicated", "--ok"), unsafe_allow_html=True)
    slots[3].markdown(kpi_html("Hazard Density", f"{density:.1f}", "per km of surveyed road", "--accent", "/km"), unsafe_allow_html=True)
    slots[4].markdown(kpi_html("Processing Speed", f"{fps_val:.1f}", fps_sub, "--violet", "fps"), unsafe_allow_html=True)


def timeline_chart(history, T, fps):
    df = pd.DataFrame(history)
    if len(df) > 600:
        df = df.iloc[:: max(1, len(df) // 600)]
    df["Time (s)"] = df["Frame"] / fps
    long = df.melt(id_vars=["Frame", "Time (s)"], value_vars=["Potholes", "Manholes"], var_name="Class", value_name="Count")
    color = alt.Color("Class:N", scale=alt.Scale(domain=["Potholes", "Manholes"], range=[T["pothole"], T["manhole"]]), legend=alt.Legend(title=None))
    base = alt.Chart(long).encode(
        x=alt.X("Time (s):Q", title="Video time (s)"),
        y=alt.Y("Count:Q", title="Unique hazards", axis=alt.Axis(tickMinStep=1)),
        color=color,
    )
    area = base.mark_area(opacity=0.12, interpolate="step-after")
    line = base.mark_line(strokeWidth=2.5, interpolate="step-after")
    return style_chart((area + line).properties(height=320), T)


def registry_table_html(records, T):
    head = "".join(f"<th>{h}</th>" for h in ["ID", "Type", "Priority", "Chainage", "Video time", "Tracked", "Peak confidence", "Size", "GPS"])
    rows = []
    for r in records:
        cc = class_color(r["cls"], T)
        pc = priority_color(r["priority"], T)
        rows.append(
            "<tr>"
            f'<td class="mono"><b>#{r["id"]}</b></td>'
            f'<td><span class="pill" style="--c:{cc}">{r["cls"].capitalize()}</span></td>'
            f'<td><span class="pill" style="--c:{pc}">{r["priority"]}</span></td>'
            f'<td class="mono">{r["chainage_m"] / 1000:.3f} km</td>'
            f'<td class="mono">{r["time_s"]:.1f}s</td>'
            f'<td class="mono">{r["frames"]} fr</td>'
            f'<td><span class="bar"><span style="width:{r["conf"] * 100:.0f}%;--c:{cc}"></span></span><span class="mono">{r["conf"]:.0%}</span></td>'
            f'<td class="mono">{r["area_pct"]:.2f}%</td>'
            f'<td class="mono">{r["lat"]:.5f}, {r["lon"]:.5f}</td>'
            "</tr>"
        )
    return f'<div class="reg-wrap"><table class="reg"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def render_empty_state():
    html(
        '<div class="steps">'
        '<div class="step"><div class="n">01</div><h4>Load footage</h4><p>Upload dash-cam video or pick a clip from the <code>tests/</code> folder in the sidebar.</p></div>'
        '<div class="step"><div class="n">02</div><h4>Detect &amp; track</h4><p>YOLO detects hazards each frame; the Kalman tracker assigns stable IDs so each pothole is counted once.</p></div>'
        '<div class="step"><div class="n">03</div><h4>Map, prioritise, export</h4><p>Get a geo-referenced hazard map, repair priorities, evidence snapshots and CSV / JSON / GeoJSON exports.</p></div>'
        "</div>"
    )


# ==================================================
# Sidebar
# ==================================================
with st.sidebar:
    html(
        '<div class="brand"><div class="brand-logo">🚧</div>'
        '<div><div class="brand-name">RoadSense AI</div><div class="brand-tag">Hazard detection · tracking · GIS</div></div></div>'
    )
    dark = st.toggle("🌙 Dark mode", value=True, key="dark_mode")

T = THEMES["dark" if dark else "light"]
inject_theme(T)

with st.sidebar:
    html('<div class="side-sec">Model & hardware</div>')
    available_models = find_available_models()
    selected_model_path = st.selectbox("Model weights", options=available_models, index=0, format_func=lambda p: Path(p).name)

    try:
        with st.spinner("Loading model…"):
            model = load_yolo_model(selected_model_path)
        model_classes = {k: v.lower() for k, v in model.names.items()}
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    gpu = torch.cuda.is_available()
    device = st.selectbox(
        "Compute device",
        options=["0", "cpu"] if gpu else ["cpu"],
        format_func=lambda d: f"GPU · {torch.cuda.get_device_name(0)}" if d == "0" else "CPU",
    )
    html(
        f'<div class="model-card"><b>{Path(selected_model_path).name}</b>'
        f'<div class="mc-row"><span>Classes</span><span>{len(model_classes)}</span></div>'
        f'<div class="mc-row"><span>Labels</span><span>{", ".join(model_classes.values())}</span></div></div>'
    )

    html('<div class="side-sec">Detection</div>')
    conf_thresh = st.slider("Confidence threshold", 0.05, 1.0, 0.15, 0.05,
                            help="0.15 recommended for high recall on subtle or shadowed potholes.")
    use_tta = st.toggle("⚡ Test-time augmentation", value=True,
                        help="Multi-scale flipped ensembling. Better recall on small/distant potholes, ~2-3x slower.")

    with st.expander("Tracking & performance"):
        iou_thresh = st.slider("IoU matching threshold", 0.1, 0.9, 0.25, 0.05,
                               help="Spatial IoU threshold for Hungarian matching.")
        min_confirm_hits = st.slider("Anti-flicker confirmation frames", 1, 5, 2,
                                     help="Consecutive frames a hazard must appear before it is counted.")
        frame_stride = st.select_slider("Frame stride", options=[1, 2, 3, 4], value=1,
                                        help="1 = every frame. 2 = every 2nd frame (2x faster).")
        ui_every = st.select_slider("Live preview refresh", options=[1, 2, 3, 5, 10], value=2,
                                    help="Update the dashboard every N processed frames. Higher = faster analysis.")

    with st.expander("Geo-reference"):
        base_lat = st.number_input("Start latitude", value=-28.753000, format="%.6f")
        base_lon = st.number_input("Start longitude", value=31.895000, format="%.6f")
        bearing = st.slider("Route bearing (°)", 0, 359, 45, help="Direction of travel, 0 = north, 90 = east.")
        speed_kmh = st.slider("Average vehicle speed (km/h)", 10, 120, 45)

    html('<div class="side-sec">Video source</div>')
    input_mode = st.radio("Input mode", ["Upload", "tests/ folder"], horizontal=True, label_visibility="collapsed")

    video_input_path = None
    input_filename = None

    if input_mode == "Upload":
        uploaded_file = st.file_uploader("Upload road video", type=["mp4", "avi", "mov", "mkv", "webm"])
        if uploaded_file is not None:
            up_key = f"{uploaded_file.name}-{uploaded_file.size}"
            # Write to disk once per file, not on every rerun
            if st.session_state.get("upload_key") != up_key:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
                tfile.write(uploaded_file.getbuffer())
                tfile.close()
                st.session_state["upload_key"] = up_key
                st.session_state["upload_path"] = tfile.name
            video_input_path = st.session_state["upload_path"]
            input_filename = uploaded_file.name
    else:
        sample_videos = find_test_videos()
        if sample_videos:
            chosen_video = st.selectbox("Sample video", options=sample_videos, format_func=lambda p: p.name)
            if chosen_video:
                video_input_path = str(chosen_video)
                input_filename = chosen_video.name
        else:
            st.warning("No sample videos found in `tests/`.")

# ==================================================
# Hero header
# ==================================================
device_label = "GPU" if device != "cpu" else "CPU"
html(
    '<div class="hero"><div class="hero-row"><div>'
    '<div class="eyebrow">Road asset inspection</div>'
    "<h1>Road Damage Detection &amp; Spatial Inspection</h1>"
    '<p class="lead">YOLO11 detection, Kalman velocity tracking and GIS mapping. Every hazard gets a stable ID, '
    "a repair priority and a map position — and is never counted twice.</p></div>"
    '<div class="chips">'
    f'<span class="chip"><span class="dot"></span>Model ready</span>'
    f'<span class="chip">🧠 {Path(selected_model_path).name}</span>'
    f'<span class="chip">🖥 {device_label}</span>'
    f'<span class="chip" style="--c:{T["accent"] if use_tta else T["faint"]}"><span class="dot"></span>TTA {"on" if use_tta else "off"}</span>'
    "</div></div></div>"
)

if not video_input_path:
    render_empty_state()
    st.write("")
    st.info("👈 Upload a road video or choose a sample in the sidebar to begin.")
    st.stop()

meta = probe_video(video_input_path)
results = st.session_state.get("results")
if results and results.get("source") != input_filename:
    results = None

# Source bar + run button
src_col, btn_col = st.columns([5, 2], vertical_alignment="center")
with src_col:
    html(
        '<div class="src-bar"><div class="src-ico">🎞️</div><div>'
        f'<div class="src-name">{input_filename}</div>'
        f'<div class="src-meta">{meta["w"]}×{meta["h"]} · {meta["fps"]:.1f} fps · {meta["frames"]} frames · {fmt_dur(meta["duration"])}'
        f' · ≈{meta["duration"] * speed_kmh / 3.6 / 1000:.2f} km route</div></div></div>'
    )
with btn_col:
    start_btn = st.button("🚀 Run inspection", type="primary", use_container_width=True)

st.write("")

# KPI row
kpi_slots = [c.empty() for c in st.columns(5)]
if results:
    render_kpis(kpi_slots, results["cumulative"], {}, results["density"], results["avg_fps"], f"Average · {results['processed']} frames")
else:
    render_kpis(kpi_slots, {}, {}, 0.0, 0.0, "Awaiting run")

st.write("")
progress_bar = st.progress(1.0 if results else 0.0)
status_slot = st.empty()
if results:
    status_slot.markdown(
        f'<div class="statusbar">✅ <span>Analysis complete</span><span>Hazards <b>{results["total"]}</b></span>'
        f'<span>Run time <b>{fmt_dur(results["elapsed"])}</b></span><span>Avg speed <b>{results["avg_fps"]:.1f} fps</b></span></div>',
        unsafe_allow_html=True,
    )

st.write("")
feed_col, chart_col = st.columns([3, 2], gap="large")
with feed_col:
    feed_title = st.empty()
    live_frame_view = st.empty()
with chart_col:
    html('<div class="panel-title"><h3>Cumulative detections</h3><span>unique hazards over time</span></div>')
    live_chart_view = st.empty()

if not start_btn:
    feed_title.markdown('<div class="panel-title"><h3>Live inspection feed</h3><span>idle</span></div>', unsafe_allow_html=True)
    if results and results.get("last_frame") is not None:
        live_frame_view.image(results["last_frame"], use_container_width=True)
    else:
        live_frame_view.markdown('<div class="placeholder"><div><div class="big">🛣️</div>Press <b>Run inspection</b> to start the live feed</div></div>', unsafe_allow_html=True)
    if results and results["history"]:
        live_chart_view.altair_chart(timeline_chart(results["history"], T, results["fps"]), use_container_width=True, theme=None)
    else:
        live_chart_view.markdown('<div class="placeholder" style="height:320px"><div><div class="big">📈</div>The timeline appears once analysis starts</div></div>', unsafe_allow_html=True)

# ==================================================
# Analysis loop
# ==================================================
if start_btn:
    st.session_state.pop("results", None)
    feed_title.markdown('<div class="panel-title"><h3>Live inspection feed</h3><span class="live">LIVE</span></div>', unsafe_allow_html=True)

    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        st.error("Could not open video.")
        st.stop()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = float(max(1, width * height))
    m_per_frame = (speed_kmh / 3.6) / orig_fps  # metres travelled per raw video frame

    temp_dir = tempfile.mkdtemp()
    temp_raw_video = Path(temp_dir) / "annotated_raw.avi"
    final_output_mp4 = Path(temp_dir) / "road_damage_tracked.mp4"

    effective_fps = max(1.0, orig_fps / frame_stride)
    out_writer = cv2.VideoWriter(str(temp_raw_video), cv2.VideoWriter_fourcc(*"MJPG"), effective_fps, (width, height))

    tracker = RoadHazardKalmanTracker(
        min_confirmation_hits=min_confirm_hits,
        max_age=int(orig_fps * 1.5),
        iou_threshold=iou_thresh,
    )

    frame_idx = 0
    processed = 0
    t_start = time.time()
    last_t = t_start
    current_fps = 0.0
    history = []
    peak_area = {}  # hazard id -> largest bbox area ratio seen
    evidence = {}  # hazard id -> (confidence, jpeg bytes)
    cumulative = {}
    preview_rgb = None
    disp_scale = min(1.0, 960 / max(1, width))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_stride != 0:
            continue
        processed += 1

        res = model.predict(source=frame, conf=conf_thresh, iou=iou_thresh, device=device, augment=use_tta, verbose=False)[0]
        detections = [
            {
                "bbox": box.xyxy[0].tolist(),
                "class_name": normalize_hazard_class(model.names.get(int(box.cls[0].item()), "")),
                "conf": float(box.conf[0].item()),
            }
            for box in res.boxes
        ]

        visible_hazards, cumulative = tracker.update(detections, frame_idx)

        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            inst = 1.0 / dt
            current_fps = inst if current_fps == 0 else 0.85 * current_fps + 0.15 * inst

        annotated = frame.copy()
        in_frame = {}
        for hz in visible_hazards:
            x1, y1, x2, y2 = [int(v) for v in hz.get_bbox()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            in_frame[hz.class_name] = in_frame.get(hz.class_name, 0) + 1

            area = (x2 - x1) * (y2 - y1) / frame_area
            peak_area[hz.id] = max(peak_area.get(hz.id, 0.0), area)

            # Keep the highest-confidence crop of each hazard as photo evidence
            conf_now = float(hz.conf)
            if conf_now >= evidence.get(hz.id, (-1.0, None))[0]:
                pw, ph = int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)
                crop = frame[max(0, y1 - ph) : min(height, y2 + ph), max(0, x1 - pw) : min(width, x2 + pw)]
                if crop.size:
                    s = 320 / max(crop.shape[:2])
                    if s < 1:
                        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    if ok:
                        evidence[hz.id] = (conf_now, buf.tobytes())

            draw_hazard(annotated, (x1, y1, x2, y2), DRAW_BGR.get(hz.class_name, (94, 197, 34)),
                        f"#{hz.id} {hz.class_name.upper()} {conf_now:.0%}")

        draw_hud(annotated, width, frame_idx, total_frames, cumulative)
        out_writer.write(annotated)

        history.append({"Frame": frame_idx, "Potholes": cumulative.get("pothole", 0), "Manholes": cumulative.get("manhole", 0)})

        if processed % ui_every == 0 or processed == 1:
            surveyed_km = frame_idx * m_per_frame / 1000
            density = sum(cumulative.values()) / surveyed_km if surveyed_km > 0.01 else 0.0
            render_kpis(kpi_slots, cumulative, in_frame, density, current_fps, f"Frame {frame_idx}/{total_frames}")

            pct = min(1.0, frame_idx / total_frames)
            progress_bar.progress(pct)
            elapsed = now - t_start
            eta = elapsed / pct * (1 - pct) if pct > 0 else 0
            status_slot.markdown(
                f'<div class="statusbar"><span class="live">PROCESSING</span><span>Progress <b>{pct:.0%}</b></span>'
                f'<span>Frame <b>{frame_idx}/{total_frames}</b></span><span>Hazards <b>{sum(cumulative.values())}</b></span>'
                f'<span>Elapsed <b>{fmt_dur(elapsed)}</b></span><span>ETA <b>{fmt_dur(eta)}</b></span></div>',
                unsafe_allow_html=True,
            )

            disp = cv2.resize(annotated, None, fx=disp_scale, fy=disp_scale, interpolation=cv2.INTER_AREA) if disp_scale < 1 else annotated
            preview_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            live_frame_view.image(preview_rgb, use_container_width=True)

        if processed % 15 == 0:
            live_chart_view.altair_chart(timeline_chart(history, T, orig_fps), use_container_width=True, theme=None)

    cap.release()
    out_writer.release()
    elapsed = time.time() - t_start

    if history:
        live_chart_view.altair_chart(timeline_chart(history, T, orig_fps), use_container_width=True, theme=None)
    progress_bar.progress(1.0)

    with st.spinner("Encoding browser-ready video…"):
        try:
            convert_to_web_mp4(temp_raw_video, final_output_mp4)
            output_video_path = str(final_output_mp4)
        except Exception as err:
            st.warning(f"MP4 encoding failed, falling back to raw AVI: {err}")
            output_video_path = str(temp_raw_video)

    # Build geo-referenced hazard registry
    records = []
    for hid, info in sorted(tracker.hazard_registry.items(), key=lambda kv: kv[1]["first_frame"]):
        cls = info["class_name"]
        chain_m = info["first_frame"] * m_per_frame
        lat, lon = offset_coords(base_lat, base_lon, chain_m, bearing)
        area = peak_area.get(hid, 0.0)
        score, prio = priority_of(cls, float(info["peak_conf"]), area)
        records.append(
            {
                "id": hid,
                "cls": cls,
                "first_frame": int(info["first_frame"]),
                "last_frame": int(info["last_frame"]),
                "frames": int(info["last_frame"] - info["first_frame"] + 1),
                "time_s": round(info["first_frame"] / orig_fps, 2),
                "conf": float(info["peak_conf"]),
                "area_pct": round(area * 100, 3),
                "score": round(score, 3),
                "priority": prio,
                "chainage_m": round(chain_m, 1),
                "lat": float(lat),
                "lon": float(lon),
            }
        )

    route_m = total_frames * m_per_frame
    cum_final = dict(tracker.cumulative_counts)
    total_unique = sum(cum_final.values())
    avg_fps = processed / elapsed if elapsed > 0 else 0.0
    density = total_unique / (route_m / 1000) if route_m > 10 else 0.0

    st.session_state["results"] = {
        "source": input_filename,
        "video_path": output_video_path,
        "records": records,
        "evidence": {hid: v[1] for hid, v in evidence.items()},
        "cumulative": cum_final,
        "total": total_unique,
        "history": history,
        "fps": orig_fps,
        "route_m": route_m,
        "density": density,
        "elapsed": elapsed,
        "avg_fps": avg_fps,
        "processed": processed,
        "resolution": f"{width}×{height}",
        "last_frame": preview_rgb,
        "settings": {
            "model": Path(selected_model_path).name, "conf": conf_thresh, "tta": use_tta,
            "iou": iou_thresh, "confirm": min_confirm_hits, "stride": frame_stride,
            "speed_kmh": speed_kmh, "bearing": bearing, "base": [base_lat, base_lon],
        },
    }
    st.rerun()

# ==================================================
# Results
# ==================================================
if results and Path(results["video_path"]).is_file():
    records = results["records"]
    st.write("")
    tab_video, tab_map, tab_analytics, tab_registry, tab_evidence = st.tabs(
        ["🎬 Inspection video", "🗺️ Hazard map", "📈 Analytics", "🧾 Registry & export", "🖼️ Evidence"]
    )
    stem = Path(results["source"]).stem

    # ---------------- Video ----------------
    with tab_video:
        v_col, s_col = st.columns([3, 2], gap="large")
        with v_col:
            with open(results["video_path"], "rb") as vf:
                v_bytes = vf.read()
            st.video(v_bytes)
            st.download_button(
                "⬇️ Download annotated video", data=v_bytes, file_name=f"inspected_{stem}.mp4",
                mime="video/mp4", type="primary", use_container_width=True,
            )
        with s_col:
            html('<div class="panel-title"><h3>Inspection summary</h3><span>strictly counted once</span></div>')
            cum = results["cumulative"]
            prio_counts = {p: sum(1 for r in records if r["priority"] == p) for p in PRIORITY_ORDER}
            rows = "".join(
                f'<div class="legend-row"><span><span class="sw" style="--c:{c}"></span>{lbl}</span><b class="mono">{n}</b></div>'
                for lbl, c, n in [
                    ("Potholes", T["pothole"], cum.get("pothole", 0)),
                    ("Manholes", T["manhole"], cum.get("manhole", 0)),
                    ("High priority", priority_color("High", T), prio_counts["High"]),
                    ("Medium priority", priority_color("Medium", T), prio_counts["Medium"]),
                    ("Low priority", priority_color("Low", T), prio_counts["Low"]),
                ]
            )
            html(f'<div class="legend">{rows}<div class="legend-row" style="border-top:1px solid var(--border);padding-top:10px">'
                 f'<span><b>Total unique hazards</b></span><b class="mono" style="font-size:1.2rem">{results["total"]}</b></div></div>')
            st.write("")
            s = results["settings"]
            html(
                '<div class="tiles">'
                f'<div class="tile"><div class="l">Route</div><div class="v">{results["route_m"] / 1000:.2f} km</div></div>'
                f'<div class="tile"><div class="l">Run time</div><div class="v">{fmt_dur(results["elapsed"])}</div></div>'
                f'<div class="tile"><div class="l">Resolution</div><div class="v">{results["resolution"]}</div></div>'
                f'<div class="tile"><div class="l">Conf · TTA</div><div class="v">{s["conf"]:.2f} · {"on" if s["tta"] else "off"}</div></div>'
                "</div>"
            )

    # ---------------- Map ----------------
    with tab_map:
        if records:
            map_df = pd.DataFrame(records)
            map_df["color"] = map_df["cls"].apply(lambda c: hex_to_rgba(class_color(c, T)))
            map_df["radius"] = 6 + map_df["score"] * 16
            map_df["label"] = map_df["cls"].str.capitalize()
            map_df["conf_txt"] = (map_df["conf"] * 100).round(0).astype(int).astype(str) + "%"
            map_df["km"] = (map_df["chainage_m"] / 1000).round(3)

            s = results["settings"]
            n_pts = 60
            route_pts = [list(offset_coords(s["base"][0], s["base"][1], d, s["bearing"]))[::-1]
                         for d in np.linspace(0, results["route_m"], n_pts)]

            m_col, l_col = st.columns([4, 1], gap="medium")
            with l_col:
                html('<div class="panel-title"><h3>Layers</h3></div>')
                show_heat = st.toggle("Density heatmap", value=False)
                show_route = st.toggle("Survey route", value=True)
                classes_on_map = st.multiselect("Show classes", sorted(map_df["cls"].unique()),
                                                default=sorted(map_df["cls"].unique()), format_func=str.capitalize)
                rows = "".join(
                    f'<div class="legend-row"><span><span class="sw" style="--c:{class_color(c, T)}"></span>{c.capitalize()}</span>'
                    f'<b class="mono">{int((map_df["cls"] == c).sum())}</b></div>'
                    for c in sorted(map_df["cls"].unique())
                )
                html(f'<div class="legend">{rows}<div style="font-size:.74rem;color:var(--muted)">Marker size = repair priority</div></div>')

            shown = map_df[map_df["cls"].isin(classes_on_map)]
            layers = []
            if show_route:
                layers.append(pdk.Layer(
                    "PathLayer", data=[{"path": route_pts}], get_path="path",
                    get_color=hex_to_rgba(T["accent"], 170), width_min_pixels=4, rounded=True,
                ))
            if show_heat and len(shown):
                layers.append(pdk.Layer(
                    "HeatmapLayer", data=shown, get_position="[lon, lat]", get_weight="score",
                    radius_pixels=50, opacity=0.55,
                ))
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=shown, get_position="[lon, lat]", get_fill_color="color",
                get_radius="radius", radius_units="pixels", stroked=True, get_line_color=[255, 255, 255, 200],
                line_width_min_pixels=1, pickable=True, auto_highlight=True,
            ))
            view = pdk.ViewState(latitude=map_df["lat"].mean(), longitude=map_df["lon"].mean(), zoom=14.5, pitch=40, bearing=0)
            deck = pdk.Deck(
                layers=layers, initial_view_state=view, map_provider="carto", map_style=T["map"],
                tooltip={
                    "html": "<b>#{id} · {label}</b><br/>Priority: {priority}<br/>Peak confidence: {conf_txt}<br/>Chainage: {km} km",
                    "style": {"backgroundColor": T["surface"], "color": T["text"], "border": f"1px solid {T['border']}",
                              "borderRadius": "10px", "fontFamily": "Inter", "fontSize": "12px"},
                },
            )
            with m_col:
                st.pydeck_chart(deck, use_container_width=True, height=560)
            st.caption("Positions are dead-reckoned from video time, the configured start point, bearing and speed. "
                       "Use the Geo-reference settings to align the route; plug in real GPS logs for survey-grade accuracy.")
        else:
            st.info("No hazards detected on this route.")

    # ---------------- Analytics ----------------
    with tab_analytics:
        if records:
            df = pd.DataFrame(records)
            df["Class"] = df["cls"].str.capitalize()
            classes = sorted(df["Class"].unique())
            class_scale = alt.Scale(domain=classes, range=[class_color(c.lower(), T) for c in classes])

            route_m = max(1.0, results["route_m"])
            seg = 100 if route_m <= 3000 else 250 if route_m <= 10000 else 500
            df["Segment"] = (df["chainage_m"] // seg * seg / 1000).round(2)
            worst = df.groupby("Segment").size().sort_values(ascending=False)

            html(
                '<div class="tiles">'
                f'<div class="tile"><div class="l">Hazards / km</div><div class="v">{results["density"]:.1f}</div></div>'
                f'<div class="tile"><div class="l">High priority</div><div class="v" style="color:var(--pothole)">{int((df["priority"] == "High").sum())}</div></div>'
                f'<div class="tile"><div class="l">Mean confidence</div><div class="v">{df["conf"].mean():.0%}</div></div>'
                f'<div class="tile"><div class="l">Worst {seg} m segment</div><div class="v">km {worst.index[0]:.2f}</div></div>'
                f'<div class="tile"><div class="l">Avg speed</div><div class="v">{results["avg_fps"]:.1f} fps</div></div>'
                "</div>"
            )
            st.write("")

            c1, c2 = st.columns([3, 2], gap="large")
            with c1:
                html(f'<div class="panel-title"><h3>Hazards per {seg} m segment</h3><span>where to send crews first</span></div>')
                seg_df = df.groupby(["Segment", "Class"]).size().reset_index(name="Count")
                bars = alt.Chart(seg_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X("Segment:O", title="Chainage (km)", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("sum(Count):Q", title="Hazards", axis=alt.Axis(tickMinStep=1)),
                    color=alt.Color("Class:N", scale=class_scale, legend=alt.Legend(title=None)),
                    tooltip=["Segment", "Class", "Count"],
                ).properties(height=300)
                st.altair_chart(style_chart(bars, T), use_container_width=True, theme=None)
            with c2:
                html('<div class="panel-title"><h3>Repair priority mix</h3><span>confidence × size</span></div>')
                pr = df["priority"].value_counts().reindex(PRIORITY_ORDER, fill_value=0).reset_index()
                pr.columns = ["Priority", "Count"]
                donut = alt.Chart(pr).mark_arc(innerRadius=75, cornerRadius=4, padAngle=0.02).encode(
                    theta="Count:Q",
                    color=alt.Color("Priority:N", sort=PRIORITY_ORDER,
                                    scale=alt.Scale(domain=PRIORITY_ORDER, range=[priority_color(p, T) for p in PRIORITY_ORDER]),
                                    legend=alt.Legend(title=None)),
                    tooltip=["Priority", "Count"],
                ).properties(height=300)
                st.altair_chart(style_chart(donut, T), use_container_width=True, theme=None)

            html('<div class="panel-title"><h3>Hazard profile along the route</h3><span>bubble size = apparent size in frame</span></div>')
            scatter = alt.Chart(df).mark_circle(opacity=0.85, stroke=T["surface"], strokeWidth=1).encode(
                x=alt.X("chainage_m:Q", title="Chainage (m)"),
                y=alt.Y("conf:Q", title="Peak confidence", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                size=alt.Size("area_pct:Q", title="Size %", scale=alt.Scale(range=[40, 600]), legend=None),
                color=alt.Color("Class:N", scale=class_scale, legend=alt.Legend(title=None)),
                tooltip=[alt.Tooltip("id:N", title="ID"), "Class", "priority",
                         alt.Tooltip("conf:Q", format=".0%", title="Confidence"),
                         alt.Tooltip("chainage_m:Q", title="Chainage (m)")],
            ).properties(height=280)
            st.altair_chart(style_chart(scatter, T), use_container_width=True, theme=None)
        else:
            st.info("No hazards to analyse.")

    # ---------------- Registry ----------------
    with tab_registry:
        if records:
            f1, f2, f3 = st.columns([2, 2, 2])
            all_cls = sorted({r["cls"] for r in records})
            sel_cls = f1.multiselect("Class", all_cls, default=all_cls, format_func=str.capitalize)
            sel_pr = f2.multiselect("Priority", PRIORITY_ORDER, default=PRIORITY_ORDER)
            sort_by = f3.selectbox("Sort by", ["Chainage", "Priority score", "Confidence", "Size"])
            key = {"Chainage": ("chainage_m", False), "Priority score": ("score", True),
                   "Confidence": ("conf", True), "Size": ("area_pct", True)}[sort_by]
            view_recs = sorted([r for r in records if r["cls"] in sel_cls and r["priority"] in sel_pr],
                               key=lambda r: r[key[0]], reverse=key[1])

            html(f'<div class="panel-title"><h3>Hazard registry</h3><span>{len(view_recs)} of {len(records)} hazards</span></div>')
            html(registry_table_html(view_recs, T))
            st.write("")

            export_df = pd.DataFrame(view_recs).rename(columns={
                "id": "hazard_id", "cls": "class", "conf": "peak_confidence", "area_pct": "peak_size_pct",
                "score": "priority_score", "time_s": "video_time_s", "frames": "frames_tracked",
            })
            geojson = {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                     "properties": {k: v for k, v in r.items() if k not in ("lat", "lon")}}
                    for r in view_recs
                ],
            }
            e1, e2, e3 = st.columns(3)
            e1.download_button("📥 CSV report", export_df.to_csv(index=False).encode("utf-8"),
                               f"road_hazards_{stem}.csv", "text/csv", use_container_width=True)
            e2.download_button("📥 JSON report", json.dumps(view_recs, indent=2, default=str).encode("utf-8"),
                               f"road_hazards_{stem}.json", "application/json", use_container_width=True)
            e3.download_button("🌍 GeoJSON (QGIS / ArcGIS)", json.dumps(geojson, indent=2, default=str).encode("utf-8"),
                               f"road_hazards_{stem}.geojson", "application/geo+json", use_container_width=True)
        else:
            st.info("No hazards recorded.")

    # ---------------- Evidence ----------------
    with tab_evidence:
        ev = results["evidence"]
        ev_recs = [r for r in records if r["id"] in ev]
        if ev_recs:
            g1, g2 = st.columns([2, 4])
            ev_cls = g1.selectbox("Filter", ["All"] + sorted({r["cls"] for r in ev_recs}), format_func=str.capitalize)
            if ev_cls != "All":
                ev_recs = [r for r in ev_recs if r["cls"] == ev_cls]
            ev_recs = sorted(ev_recs, key=lambda r: r["score"], reverse=True)[:60]
            html(f'<div class="panel-title"><h3>Best-confidence snapshot per hazard</h3><span>ranked by repair priority · {len(ev_recs)} shown</span></div>')
            per_row = 5
            for i in range(0, len(ev_recs), per_row):
                cols = st.columns(per_row)
                for col, r in zip(cols, ev_recs[i : i + per_row]):
                    with col:
                        st.image(ev[r["id"]], use_container_width=True)
                        html(
                            f'<div class="ev-cap"><span><b>#{r["id"]}</b> · {r["cls"].capitalize()}</span>'
                            f'<span class="pill" style="--c:{priority_color(r["priority"], T)}">{r["priority"]}</span></div>'
                        )
        else:
            st.info("No evidence snapshots captured.")