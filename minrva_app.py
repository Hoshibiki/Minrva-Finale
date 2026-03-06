# minrva_app.py — Minrva Focus Estimator (Exhibition Edition)
import streamlit as st
import cv2
import numpy as np
import math
import os
import time
import urllib.request
import threading
from pathlib import Path

from ultralytics import YOLO
import mediapipe as mp
import platform
import pyttsx3
import requests

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Minrva",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #0a0c10;
    --surface:   #111318;
    --border:    #1e2230;
    --accent:    #7DF9AA;
    --accent2:   #5BC8F5;
    --warn:      #FFD166;
    --danger:    #FF5C5C;
    --text:      #E8EAF0;
    --muted:     #5a6070;
    --radius:    12px;
}

/* ── Global reset ── */
html, body {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}
/* App container must be transparent so bg-layer shows through */
[data-testid="stMain"],
[data-testid="stMain"] > div {
    background: transparent !important;
}
/* Solid fallback color on the root only, overridden by bg-layer when active */
[data-testid="stAppViewContainer"] {
    color: var(--text) !important;
}
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    z-index: 100 !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
/* Keep toolbar visible so native sidebar control remains accessible */
[data-testid="collapsedControl"] {
    display: block !important;
    z-index: 1000 !important;
    opacity: 1 !important;
    visibility: visible !important;
}
[data-testid="collapsedControl"] button {
    background: #111318 !important;
    border: 1px solid #7DF9AA !important;
    border-radius: 0 8px 8px 0 !important;
}
[data-testid="collapsedControl"] button svg {
    color: #7DF9AA !important;
    fill: #7DF9AA !important;
    opacity: 1 !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
}
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* ── Headings ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.05em;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    box-shadow: 0 0 12px rgba(125,249,170,0.15) !important;
}

/* ── Primary action button ── */
.btn-primary > button {
    background: var(--accent) !important;
    color: #0a0c10 !important;
    border: none !important;
    font-weight: 700 !important;
}
.btn-primary > button:hover {
    box-shadow: 0 0 20px rgba(125,249,170,0.4) !important;
    color: #0a0c10 !important;
}

/* ── Stop button ── */
.btn-stop > button {
    border-color: var(--danger) !important;
    color: var(--danger) !important;
}
.btn-stop > button:hover {
    background: rgba(255,92,92,0.1) !important;
    box-shadow: 0 0 12px rgba(255,92,92,0.2) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] .rc-slider-track { background: var(--accent) !important; }
[data-testid="stSlider"] .rc-slider-handle { border-color: var(--accent) !important; background: var(--accent) !important; }

/* ── Selectbox / Checkbox ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stCheckbox"] label { font-family: 'Space Mono', monospace !important; font-size: 12px !important; }
[data-testid="stSelectbox"] > div > div {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stExpander"] summary { font-family: 'Space Mono', monospace !important; font-size: 12px !important; color: var(--muted) !important; }

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: var(--radius) !important; }

/* ── Metric cards ── */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 20px;
    text-align: center;
}
.metric-card .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-card .metric-label {
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
}

/* ── Live score ring ── */
.score-ring-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 12px;
}
.score-ring-container svg { filter: drop-shadow(0 0 18px currentColor); }
.score-value-text {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    margin-top: 8px;
    line-height: 1;
}
.score-status-text {
    font-size: 11px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    text-align: center;
    margin-top: 4px;
}

/* ── Session summary ── */
.summary-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px;
    margin-top: 16px;
}
.summary-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 8px;
}
.summary-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 20px;
}

/* ── Sidebar section label ── */
.sidebar-section {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 12px 0 6px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 10px;
}

/* ── Status pill ── */
.status-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Top wordmark ── */
.wordmark {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    color: var(--accent);
    line-height: 1;
}
.wordmark span {
    color: var(--muted);
    font-weight: 400;
    font-size: 0.9rem;
    margin-left: 6px;
    vertical-align: middle;
}

/* ── Camera feed border ── */
[data-testid="stImage"] img {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}

/* ── Notification toast ── */
.focus-toast {
    background: linear-gradient(135deg, #FF5C5C22, #FF5C5C11);
    border: 1px solid var(--danger);
    border-radius: 12px;
    padding: 16px 20px;
    color: var(--danger);
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-align: center;
    animation: pulse-border 1s ease-in-out infinite;
}
@keyframes pulse-border {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,92,92,0.3); }
    50% { box-shadow: 0 0 0 6px rgba(255,92,92,0); }
}

/* ── Sidebar visibility — driven by session state via CSS class on body ── */
[data-testid="stSidebar"] {
    transition: transform 0.3s ease !important;
}
/* Hide native Streamlit collapse arrow */
/* [data-testid="collapsedControl"] { display: none !important; } */
/* Floating re-open tab — only visible when sidebar is hidden */
.sidebar-reopen-tab {
    position: fixed;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    z-index: 99999;
    background: #111318;
    border: 1px solid #7DF9AA;
    border-left: none;
    border-radius: 0 8px 8px 0;
    color: #7DF9AA;
    cursor: pointer;
    padding: 10px 7px;
    box-shadow: 4px 0 16px rgba(125,249,170,0.2);
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    writing-mode: vertical-rl;
    letter-spacing: 0.1em;
}
.calib-status {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    line-height: 1.7;
}
/* ── Background image layer ── */
.bg-layer {
    position: fixed;
    inset: 0;
    z-index: -100;
    pointer-events: none;
    background-size: cover;
    background-position: center;
    filter: blur(8px) brightness(0.38) saturate(0.85);
    transform: scale(1.1);
}
/* ── Background video layer ── */
.bg-video {
    position: fixed;
    inset: 0;
    z-index: -100;
    width: 100%;
    height: 100%;
    overflow: hidden;
    pointer-events: none;
}
.bg-video video {
    position: absolute;
    top: 50%;
    left: 50%;
    min-width: 110%;
    min-height: 110%;
    transform: translate(-50%, -50%) scale(1.1);
    object-fit: cover;
    filter: blur(8px) brightness(0.38) saturate(0.85);
}
/* All Streamlit content layers sit above the bg */
[data-testid="stSidebar"],
[data-testid="stMain"],
.block-container {
    position: relative;
    z-index: 1;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER: render focus score as SVG arc
# ─────────────────────────────────────────────
def render_score_widget(score: float, distraction_count: int):
    if score >= 0.7:
        color = "#7DF9AA"
        status = "FOCUSED"
    elif score >= 0.35:
        color = "#FFD166"
        status = "LOW FOCUS"
    else:
        color = "#FF5C5C"
        status = "DISTRACTED"

    # Arc parameters
    r = 52
    cx = cy = 68
    circumference = 2 * math.pi * r
    progress = circumference * score
    gap = circumference - progress

    arc_svg = f"""
    <div class="score-ring-container">
        <svg width="136" height="136" viewBox="0 0 136 136" style="color:{color}">
            <circle cx="{cx}" cy="{cy}" r="{r}"
                fill="none" stroke="#1e2230" stroke-width="10"/>
            <circle cx="{cx}" cy="{cy}" r="{r}"
                fill="none" stroke="{color}" stroke-width="10"
                stroke-linecap="round"
                stroke-dasharray="{progress:.2f} {gap:.2f}"
                transform="rotate(-90 {cx} {cy})"/>
        </svg>
        <div class="score-value-text" style="color:{color}">{score:.2f}</div>
        <div class="score-status-text" style="color:{color}">{status}</div>
    </div>
    """

    distraction_html = f"""
    <div class="metric-card" style="margin-top:12px;">
        <div class="metric-value" style="color:#FF5C5C">{distraction_count}</div>
        <div class="metric-label">Distractions</div>
    </div>
    """

    return arc_svg + distraction_html


def render_score_graph(history: list):
    """Render a compact SVG sparkline of recent focus scores."""
    if len(history) < 2:
        return ""
    W, H = 220, 70
    pad = 8
    n = len(history)
    xs = [pad + (i / (n - 1)) * (W - 2 * pad) for i in range(n)]
    ys = [pad + (1.0 - v) * (H - 2 * pad) for v in history]

    # Build polyline points
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    # Filled area path
    area_path = f"M {xs[0]:.1f},{H - pad} " + " ".join(f"L {x:.1f},{y:.1f}" for x, y in zip(xs, ys)) + f" L {xs[-1]:.1f},{H - pad} Z"

    # Color by latest score
    latest = history[-1]
    color = "#7DF9AA" if latest >= 0.7 else "#FFD166" if latest >= 0.35 else "#FF5C5C"

    # Threshold line at 0.5
    thresh_y = pad + (1.0 - 0.5) * (H - 2 * pad)

    return f"""
    <div style="margin-top:12px;">
        <div style="font-family:'Space Mono',monospace;font-size:9px;color:#5a6070;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px">Focus History</div>
        <svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="display:block;overflow:visible">
            <defs>
                <linearGradient id="areafill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="{color}" stop-opacity="0.3"/>
                    <stop offset="100%" stop-color="{color}" stop-opacity="0.02"/>
                </linearGradient>
            </defs>
            <line x1="{pad}" y1="{thresh_y:.1f}" x2="{W-pad}" y2="{thresh_y:.1f}"
                  stroke="#1e2230" stroke-width="1" stroke-dasharray="3,3"/>
            <path d="{area_path}" fill="url(#areafill)"/>
            <polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.8"
                      stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="3" fill="{color}"/>
        </svg>
    </div>
    """


def render_session_summary(avg_score, total_frames, distraction_count, session_duration):
    if avg_score >= 0.7:
        grade = "A"
        grade_color = "#7DF9AA"
        verdict = "Great session — you stayed consistently on task."
    elif avg_score >= 0.5:
        grade = "B"
        grade_color = "#5BC8F5"
        verdict = "Solid effort. A few slips, but mostly focused."
    elif avg_score >= 0.35:
        grade = "C"
        grade_color = "#FFD166"
        verdict = "Moderate focus. Consider shorter, deeper sessions."
    else:
        grade = "D"
        grade_color = "#FF5C5C"
        verdict = "Rough session. Try minimizing distractions next time."

    mins = int(session_duration // 60)
    secs = int(session_duration % 60)

    return f"""
    <div class="summary-card">
        <div class="summary-title">Session Complete</div>
        <p style="color:#5a6070;font-family:'Space Mono',monospace;font-size:12px;margin:0">{verdict}</p>
        <div class="summary-grid">
            <div class="metric-card">
                <div class="metric-value" style="color:{grade_color}">{grade}</div>
                <div class="metric-label">Focus Grade</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color:#5BC8F5">{avg_score:.2f}</div>
                <div class="metric-label">Avg Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color:#FF5C5C">{distraction_count}</div>
                <div class="metric-label">Distractions</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color:#7DF9AA">{total_frames}</div>
                <div class="metric-label">Frames Analyzed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color:#FFD166">{mins}m {secs:02d}s</div>
                <div class="metric-label">Duration</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color:#b0b8cc">{(distraction_count / max(mins, 1)):.1f}</div>
                <div class="metric-label">Distractions/Min</div>
            </div>
        </div>
    </div>
    """


# ─────────────────────────────────────────────
# TTS
# ─────────────────────────────────────────────


def speak_focus():
    def _speak():
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            for voice in voices:
                if 'male' in voice.name.lower() or 'david' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            engine.say("FOCUS")
            engine.runAndWait()
            engine.stop()
        except Exception:
            pass
    threading.Thread(target=_speak, daemon=True).start()


def send_ntfy(topic: str, message: str, title: str = "Minrva Focus Alert"):
    if not topic:
        return
    try:
        requests.post(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": "high", "Tags": "warning"},
            timeout=3,
        )
    except Exception:
        pass


# ─────────────────────────────────────────────
# MODEL CACHING
# ─────────────────────────────────────────────
TASK_MODELS_DIR = Path(__file__).parent / "models"
TASK_MODELS = {
    "face": (
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    ),
    "hand": (
        "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    ),
    "pose": (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    ),
}


class LandmarkListAdapter:
    def __init__(self, landmarks):
        self.landmark = landmarks


def ensure_task_model(model_key):
    filename, url = TASK_MODELS[model_key]
    TASK_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = TASK_MODELS_DIR / filename
    if not model_path.exists():
        urllib.request.urlretrieve(url, model_path)
    return str(model_path)


@st.cache_resource
def load_models(mp_conf, use_yolo_flag):
    face_landmarker = hand_landmarker = pose_landmarker = None
    try:
        face_model = ensure_task_model("face")
        hand_model = ensure_task_model("hand")
        pose_model  = ensure_task_model("pose")

        face_options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=face_model),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hand_options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=hand_model),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=mp_conf,
            min_hand_presence_confidence=mp_conf,
            min_tracking_confidence=0.5,
        )
        pose_options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=pose_model),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_options)
        hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)
        pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)
    except Exception as e:
        st.error(f"MediaPipe init failed: {e}")

    yolo = None
    if use_yolo_flag:
        try:
            yolo = YOLO("yolov8s.pt")
        except Exception as e:
            st.warning(f"YOLO init failed: {e}")
    return face_landmarker, hand_landmarker, pose_landmarker, yolo


# ─────────────────────────────────────────────
# GEOMETRY / SCORING FUNCTIONS  (unchanged logic)
# ─────────────────────────────────────────────
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
], dtype=np.float64)

FACEMESH_IDX = {"nose_tip": 1, "chin": 152, "left_eye": 33, "right_eye": 263, "left_mouth": 61, "right_mouth": 291}
EAR_L_P = [33, 160, 158, 133, 153, 144]
EAR_R_P = [362, 385, 387, 263, 373, 380]


def to_numpy(x):
    try: return x.cpu().numpy()
    except Exception:
        try: return x.numpy()
        except Exception: return np.array(x)


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy >= 1e-6:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.degrees([x, y, z])


def norm_angle_deg(a):
    a = ((a + 180.0) % 360.0) - 180.0
    if a > 90: a -= 180
    if a < -90: a += 180
    return a


def estimate_head_angles_solvepnp(flm, img_shape):
    h, w = img_shape[:2]
    try:
        lm = flm.landmark
        image_points = np.array([
            (lm[FACEMESH_IDX["nose_tip"]].x * w, lm[FACEMESH_IDX["nose_tip"]].y * h),
            (lm[FACEMESH_IDX["chin"]].x * w,     lm[FACEMESH_IDX["chin"]].y * h),
            (lm[FACEMESH_IDX["left_eye"]].x * w, lm[FACEMESH_IDX["left_eye"]].y * h),
            (lm[FACEMESH_IDX["right_eye"]].x * w,lm[FACEMESH_IDX["right_eye"]].y * h),
            (lm[FACEMESH_IDX["left_mouth"]].x * w,lm[FACEMESH_IDX["left_mouth"]].y * h),
            (lm[FACEMESH_IDX["right_mouth"]].x * w,lm[FACEMESH_IDX["right_mouth"]].y * h),
        ], dtype=np.float64)
        focal_length = w
        camera_matrix = np.array([[focal_length,0,w/2],[0,focal_length,h/2],[0,0,1]], dtype=np.float64)
        dist_coeffs   = np.zeros((4, 1))
        success, rvec, tvec = cv2.solvePnP(MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success: return 0.0, 0.0, False
        R, _ = cv2.Rodrigues(rvec)
        pitch_deg, yaw_deg, _ = rotationMatrixToEulerAngles(R)
        return norm_angle_deg(yaw_deg), norm_angle_deg(pitch_deg), True
    except Exception:
        return 0.0, 0.0, False


def head_direction_score(yaw_rel, pitch_rel, face_detected, head_sens=1.0):
    if not face_detected: return 0.0
    YAW_MAX, PITCH_MAX = 35.0, 25.0
    DEAD_YAW, DEAD_PITCH = 12.0, 10.0   # degrees of free movement before any penalty
    base_yaw   = st.session_state.get("base_yaw", 0.0)
    base_pitch = st.session_state.get("base_pitch", 0.0)
    away_yaw   = st.session_state.get("away_yaw", None)
    away_pitch = st.session_state.get("away_pitch", None)
    yaw_rel_c   = abs(yaw_rel - base_yaw)
    pitch_rel_c = abs(pitch_rel - base_pitch)
    # Apply dead zone — no penalty within ±DEAD degrees of baseline
    yaw_eff   = max(0.0, yaw_rel_c - DEAD_YAW)
    pitch_eff = max(0.0, pitch_rel_c - DEAD_PITCH)
    yaw_range   = abs(away_yaw - base_yaw - DEAD_YAW + 1e-5)   if away_yaw   else (YAW_MAX   - DEAD_YAW)
    pitch_range = abs(away_pitch - base_pitch - DEAD_PITCH + 1e-5) if away_pitch else (PITCH_MAX - DEAD_PITCH)
    yaw_ratio   = np.clip(yaw_eff   / max(yaw_range,   1.0), 0, 1)
    pitch_ratio = np.clip(pitch_eff / max(pitch_range, 1.0), 0, 1)
    return float(np.clip((1.0 - (0.6*yaw_ratio + 0.4*pitch_ratio)) * head_sens, 0, 1))


def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_ear(eye_points):
    if len(eye_points) < 6: return 0.0
    p1,p2,p3,p4,p5,p6 = eye_points
    A = dist(p2,p6); B = dist(p3,p5); C = dist(p1,p4)
    return 0.0 if C < 1e-6 else (A+B)/(2.0*C)


def compute_ear_score(face_landmarks, h, w):
    if face_landmarks is None: return 1.0
    lm = face_landmarks.landmark
    def gc(i): return (lm[i].x, lm[i].y)
    left_ear  = compute_ear([gc(i) for i in EAR_L_P])
    right_ear = compute_ear([gc(i) for i in EAR_R_P])
    avg_ear = (left_ear + right_ear) / 2.0
    # Store raw for diagnostics
    st.session_state["_diag_ear_left"]  = left_ear
    st.session_state["_diag_ear_right"] = right_ear
    st.session_state["_diag_ear_avg"]   = avg_ear
    # Thresholds calibrated from real measurements:
    # open eye avg ~0.365, closed eye avg ~0.035
    EAR_OPEN, EAR_CLOSED = 0.30, 0.08
    if "prev_ear" not in st.session_state: st.session_state["prev_ear"] = avg_ear
    if "blink_counter" not in st.session_state: st.session_state["blink_counter"] = 0
    if "total_frames" not in st.session_state: st.session_state["total_frames"] = 0
    st.session_state["total_frames"] += 1
    if st.session_state["prev_ear"] > EAR_OPEN and avg_ear < EAR_CLOSED:
        st.session_state["blink_counter"] += 1
    st.session_state["prev_ear"] = avg_ear
    if avg_ear >= EAR_OPEN: return 1.0
    if avg_ear <= EAR_CLOSED: return 0.0
    return float(np.clip((avg_ear - EAR_CLOSED) / (EAR_OPEN - EAR_CLOSED), 0, 1))


def compute_slouch_score(pose_landmarks, h, w):
    if pose_landmarks is None: return 1.0
    lm = pose_landmarks.landmark
    try:
        nose_y     = lm[0].y
        shoulder_y = (lm[11].y + lm[12].y) / 2.0
        gap        = shoulder_y - nose_y
        # Store raw for diagnostics
        st.session_state["_diag_nose_y"]     = nose_y
        st.session_state["_diag_shoulder_y"] = shoulder_y
        st.session_state["_diag_gap"]        = gap
        # Thresholds calibrated from real measurements:
        # straight sitting nose_y ~0.37, slouching nose_y ~0.48
        # Camera is positioned such that nose never goes above 0.37 normally
        IDEAL_NOSE = 0.42   # score=1.0 at or above this (upright)
        BAD_NOSE   = 0.55   # score=0.0 at or below this (heavy slouch)
        score = float(np.clip(1.0 - (nose_y - IDEAL_NOSE) / (BAD_NOSE - IDEAL_NOSE), 0, 1))
        return score
    except Exception:
        return 1.0


def compute_emotion_score(face_landmarks, h, w):
    if face_landmarks is None: return 1.0
    lm = face_landmarks.landmark
    try:
        upper_lip = (lm[0].x*w,  lm[0].y*h)   # top of upper lip
        lower_lip = (lm[17].x*w, lm[17].y*h)  # bottom of lower lip
        nose_tip  = (lm[FACEMESH_IDX["nose_tip"]].x*w, lm[FACEMESH_IDX["nose_tip"]].y*h)
        chin      = (lm[FACEMESH_IDX["chin"]].x*w, lm[FACEMESH_IDX["chin"]].y*h)
        face_h    = dist(nose_tip, chin)
        if face_h < 1e-6: return 1.0
        r = dist(upper_lip, lower_lip) / face_h
        if r <= 0.1: return 1.0
        if r >= 0.3: return 0.0
        return float(np.clip(1.0 - (r - 0.1) / 0.2, 0, 1))
    except Exception: return 1.0


def compute_book_engagement(yolo_boxes, wrists):
    books = [box for (label,conf,box) in yolo_boxes if "book" in label.lower()]
    if not books: return 0.0, 0.0
    if not wrists: return 1.0, 0.0
    scores = []
    for (x1,y1,x2,y2) in books:
        bc = ((x1+x2)/2, (y1+y2)/2)
        bs = max(x2-x1, y2-y1)
        for (wx,wy) in wrists:
            nd = dist((wx,wy), bc) / bs if bs > 0 else 99
            if nd < 1.5: scores.append(1.0 - nd/1.5)
    return 1.0, float(np.clip(max(scores), 0, 1)) if scores else 0.0


def compute_hand_activity(prev_landmarks, curr_landmarks):
    if prev_landmarks is None or curr_landmarks is None: return 0.0
    prev_pts = [(lm.x, lm.y) for lm in prev_landmarks.landmark]
    curr_pts = [(lm.x, lm.y) for lm in curr_landmarks.landmark]
    avg_mov  = np.mean([dist(p,c) for p,c in zip(prev_pts,curr_pts)])
    LOW, HIGH = 0.005, 0.05
    if avg_mov >= HIGH: return 1.0
    if avg_mov <= LOW:  return 0.0
    return float(np.clip((avg_mov - LOW) / (HIGH - LOW), 0, 1))


def compute_movement_frequency(prev_center, curr_center):
    if prev_center is None or curr_center is None: return 0.5
    m = dist(prev_center, curr_center)
    if m >= 50: return 1.0
    if m <= 5:  return 0.0
    return float(np.clip((m - 5) / 45, 0, 1))


def compute_gaze_switch_frequency(gaze_history):
    if len(gaze_history) < 2: return 0.5
    switches = sum(1 for i in range(1, len(gaze_history)) if gaze_history[i] != gaze_history[i-1])
    return float(np.clip(switches / 3, 0, 1))


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
defaults = {
    "base_yaw": 0.0, "base_pitch": 0.0,
    "away_yaw": None, "away_pitch": None,
    "prev_person_center": None,
    "gaze_history": [],
    "last_notification_time": 0,
    "notification_cooldown": 5,
    "distraction_count": 0,
    "current_focus_score": 1.0,
    "phone_timer": 0,
    "prev_hand_landmarks": None,
    "prev_wrists": [],
    "blink_counter": 0,
    "total_frames": 0,
    "prev_ear": 0.3,
    "session_running": False,
    "session_data": [],
    "session_start": None,
    "unfocused_since": None,
    "grace_period": 3.0,
    "score_history": [],
    "raw_score_buffer": [],    # last 5 raw scores for smoothing
    "no_face_since": None,     # timestamp when face was last lost
    "bg_choice": "None",
    "sidebar_visible": True,
    "student_name": "",        # shown in ntfy alerts so teacher knows who
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Force-initialize keys that may be missing from older session states
for _k, _v in [
    ("auto_calib_buffer", []),
    ("auto_calib_done",   False),
    ("no_face_since",     None),
    ("last_head_score",   0.5),
    ("raw_score_buffer",  []),
    ("_diag_ear_left",    0.0),
    ("_diag_ear_right",   0.0),
    ("_diag_ear_avg",     0.0),
    ("_diag_nose_y",      0.0),
    ("_diag_shoulder_y",  0.0),
    ("_diag_gap",         0.0),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────
# NAME GATE — must enter name before app loads
# ─────────────────────────────────────────────
if not st.session_state["student_name"]:
    st.markdown("""
    <style>
    /* Hide sidebar entirely on the name screen */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stMain"] { margin-left: 0 !important; }
    .name-gate {
        max-width: 420px;
        margin: 8vh auto 0;
        text-align: center;
    }
    .name-gate-owl {
        font-size: 4rem;
        margin-bottom: 8px;
        display: block;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50%       { transform: translateY(-10px); }
    }
    .name-gate-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        color: #7DF9AA;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .name-gate-sub {
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        color: #5a6070;
        margin: 8px 0 32px;
        letter-spacing: 0.06em;
    }
    /* Style the text input on this screen */
    .name-gate-input input {
        background: #111318 !important;
        border: 1px solid #1e2230 !important;
        border-radius: 10px !important;
        color: #E8EAF0 !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 1rem !important;
        text-align: center !important;
        padding: 14px !important;
        transition: border-color 0.2s !important;
    }
    .name-gate-input input:focus {
        border-color: #7DF9AA !important;
        box-shadow: 0 0 0 2px rgba(125,249,170,0.15) !important;
    }
    .name-gate-input input::placeholder { color: #3a4050 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="name-gate">', unsafe_allow_html=True)
    st.markdown('<span class="name-gate-owl">🦉</span>', unsafe_allow_html=True)
    st.markdown('<p class="name-gate-title">Minrva</p>', unsafe_allow_html=True)
    st.markdown('<p class="name-gate-sub">FOCUS ESTIMATOR · EXHIBITION BUILD</p>', unsafe_allow_html=True)

    st.markdown('<div class="name-gate-input">', unsafe_allow_html=True)
    entered_name = st.text_input(
        "Your name",
        placeholder="Enter your name to begin...",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
        confirm = st.button("Enter Minrva  →", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if confirm:
        if entered_name.strip():
            st.session_state["student_name"] = entered_name.strip()
            st.rerun()
        else:
            st.markdown('<p style="font-family:\'Space Mono\',monospace;font-size:11px;color:#FF5C5C;text-align:center;margin-top:8px">Please enter your name first.</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()   # nothing else renders until name is confirmed

# Keep sidebar visible by default on every rerun.
# The custom hide CSS/state can get "stuck" and make the sidebar appear missing.
st.session_state["sidebar_visible"] = True


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
# BACKGROUND PRESETS  (global — used by sidebar and bg layer injection)
# ─────────────────────────────────────────────
BG_PRESETS = {
    "None":               {"url": None,                               "type": None},
    "Genius Territory":   {"url": "https://i.imgur.com/SK2iLhn.jpg", "type": "image"},
    "Warm Tea":           {"url": "https://i.imgur.com/XhbuwaW.jpg", "type": "image"},
    "MY WIFE":            {"url": "https://i.imgur.com/hzssvaQ.mp4", "type": "video"},
    "My Wife ver. 2":     {"url": "https://i.imgur.com/zawPm6q.mp4", "type": "video"},
    "A Heartfelt Smile":  {"url": "https://i.imgur.com/30R0fO1.mp4", "type": "video"},
    "Catz Chaos":         {"url": "https://i.imgur.com/u0QUi2H.jpg", "type": "image"},
    "Gangsta Cat":        {"url": "https://i.imgur.com/JjgbEDN.jpg", "type": "image"},
    "Mr. Worldwide Cat":  {"url": "https://i.imgur.com/jL2rnXF.jpg", "type": "image"},
    "Solitude":           {"url": "https://i.imgur.com/cHNHFyi.jpg", "type": "image"},
    "OST 179":            {"url": "https://i.imgur.com/RGtRqh6.mp4", "type": "video"},
}

with st.sidebar:
    # ── Wordmark + hide button ──
    st.markdown('<div class="wordmark">🦉 Minrva</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#5a6070;font-family:\'Space Mono\',monospace;font-size:11px;margin-top:4px;margin-bottom:20px">Focus Estimator · Exhibition Build</p>', unsafe_allow_html=True)

    # ── Mode ──
    st.markdown('<div class="sidebar-section">Activity Mode</div>', unsafe_allow_html=True)
    user_mode = st.selectbox("Activity type", ["Static", "Dynamic"], label_visibility="collapsed")
    mode = None
    if user_mode == "Static":
        mode = st.selectbox("Study mode", ["Digital", "Offline", "Hybrid"], label_visibility="collapsed")

    # ── Detection ──
    st.markdown('<div class="sidebar-section">Detection</div>', unsafe_allow_html=True)
    use_yolo = st.checkbox("YOLO (phone & object detection)", True)
    simulate_phone = st.checkbox("Simulate phone present", False)
    head_sensitivity = st.slider("Head sensitivity", 0.0, 1.0, 1.0, 0.05)
    mp_hand_conf     = st.slider("Hand detection confidence", 0.2, 0.9, 0.5, 0.05)
    yolo_conf        = st.slider("YOLO confidence threshold", 0.05, 0.8, 0.15, 0.05)
    phone_penalty    = st.slider("Phone penalty", 0.0, 1.0, 0.7, 0.05)
    min_area_ratio   = st.slider("Min person area ratio", 0.01, 0.30, 0.05, 0.01)

    # ── Alerts ──
    st.markdown('<div class="sidebar-section">Alerts</div>', unsafe_allow_html=True)
    enable_notifications = st.checkbox("Enable focus alerts", True)
    focus_threshold      = st.slider("Alert threshold", 0.0, 1.0, 0.5, 0.05)
    grace_period         = st.slider("Grace period (sec)", 1.0, 10.0, 3.0, 0.5)
    st.session_state["grace_period"] = grace_period
    enable_tts           = st.checkbox("Voice alert (TTS)", True)

    # ── ntfy.sh ──
    st.markdown('<div class="sidebar-section">Remote Monitoring</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="calib-status">Logged in as: <span style="color:#7DF9AA;font-weight:700">{st.session_state["student_name"]}</span></div>',
        unsafe_allow_html=True
    )
    if st.button("← Change name", use_container_width=True):
        st.session_state["student_name"] = ""
        st.rerun()
    ntfy_topic = "minrva-herta-elaina"
    st.markdown(
        f'<p style="font-family:\'Space Mono\',monospace;font-size:10px;color:#5a6070">Alerts → ntfy.sh/{ntfy_topic}</p>',
        unsafe_allow_html=True
    )

    # ── Calibration ──
    st.markdown('<div class="sidebar-section">Head Calibration</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✦ Neutral", use_container_width=True):
            st.session_state["calibrate_neutral"] = True
    with c2:
        if st.button("✦ Away", use_container_width=True):
            st.session_state["calibrate_away"] = True
    if st.button("Reset calibration", use_container_width=True):
        st.session_state.update({"base_yaw": 0.0, "base_pitch": 0.0, "away_yaw": None, "away_pitch": None})
        st.success("Calibration reset.")

    calib_away_str = f"{st.session_state['away_yaw']:.1f}°" if st.session_state['away_yaw'] else "—"
    st.markdown(f"""
    <div class="calib-status">
        Neutral  Yaw: {st.session_state['base_yaw']:.1f}° · Pitch: {st.session_state['base_pitch']:.1f}°<br>
        Away     Yaw: {calib_away_str}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Session</div>', unsafe_allow_html=True)
    if st.button("Reset distractions", use_container_width=True):
        st.session_state["distraction_count"] = 0

    # ── Background ──
    st.markdown('<div class="sidebar-section">Background</div>', unsafe_allow_html=True)
    bg_choice  = st.selectbox(
        "Background",
        list(BG_PRESETS.keys()),
        index=list(BG_PRESETS.keys()).index(st.session_state["bg_choice"]),
        key="bg_choice",
        label_visibility="collapsed"
    )


# ─────────────────────────────────────────────
# BACKGROUND LAYER  (reads from session_state — stable across reruns)
# ─────────────────────────────────────────────
_bg_preset = BG_PRESETS.get(st.session_state.get("bg_choice", "None"), {"url": None, "type": None})
_bg_url    = _bg_preset["url"]
_bg_type   = _bg_preset["type"]

if _bg_url and _bg_type == "image":
    st.markdown(
        f'<div class="bg-layer" style="background-image: url(\'{_bg_url}\');"></div>',
        unsafe_allow_html=True
    )
elif _bg_url and _bg_type == "video":
    st.markdown(f"""
    <div class="bg-video">
        <video autoplay loop muted playsinline>
            <source src="{_bg_url}" type="video/mp4">
        </video>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
face_landmarker, hand_landmarker, pose_landmarker, yolo_model = load_models(mp_hand_conf, use_yolo)


# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────

# Top header row
header_left, header_right = st.columns([3, 1])
with header_left:
    st.markdown('<h1 style="margin-bottom:0;font-size:2rem">Live Monitor</h1>', unsafe_allow_html=True)
    mode_label = mode if mode else user_mode
    st.markdown(f'<p style="color:#5a6070;font-family:\'Space Mono\',monospace;font-size:11px;margin-top:2px">Mode: {mode_label} &nbsp;·&nbsp; Session frames: {st.session_state["total_frames"]}</p>', unsafe_allow_html=True)

# Main content: feed | score panel
feed_col, score_col = st.columns([3, 1])

with feed_col:
    stframe = st.empty()

with score_col:
    score_display   = st.empty()
    alert_display   = st.empty()
    graph_display   = st.empty()   # rolling focus graph

# Controls row
ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 4])
with ctrl1:
    st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
    start_button = st.button("▶  Start", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with ctrl2:
    st.markdown('<div class="btn-stop">', unsafe_allow_html=True)
    stop_button = st.button("■  Stop", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Session summary placeholder
summary_placeholder = st.empty()

# Debug expander (collapsed by default)
debug_expander = st.expander("🔬 Debug telemetry", expanded=False)
debug_placeholder = debug_expander.empty()

# Initial score display
score_display.markdown(
    render_score_widget(st.session_state["current_focus_score"], st.session_state["distraction_count"]),
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
if start_button:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
    else:
        st.session_state["session_running"] = True
        st.session_state["session_start"]   = time.time()
        st.session_state["session_data"]    = []
        st.session_state["distraction_count"] = 0
        st.session_state["total_frames"]    = 0
        st.session_state["blink_counter"]   = 0
        st.session_state["score_history"]    = []
        st.session_state["raw_score_buffer"] = []
        st.session_state["unfocused_since"]  = None
        st.session_state["no_face_since"]    = None
        st.session_state["last_head_score"]  = 0.5

        total_score = 0.0
        frame_count = 0

        while True:
            ret, frame = cap.read()
            now_time = time.time()  # available throughout entire frame loop
            if not ret:
                st.warning("Frame read failed.")
                break

            h, w = frame.shape[:2]
            img_area = h * w
            out = frame.copy()

            # ── YOLO ──
            yolo_boxes = []
            yolo_labels = []
            if use_yolo and yolo_model:
                try:
                    results = yolo_model(frame, conf=yolo_conf, verbose=False)
                    for det in results:
                        boxes   = to_numpy(det.boxes.xyxy)
                        confs   = to_numpy(det.boxes.conf)
                        classes = to_numpy(det.boxes.cls).astype(int)
                        for box, conf, cls in zip(boxes, confs, classes):
                            label = yolo_model.names[cls]
                            x1,y1,x2,y2 = map(int, box)
                            yolo_boxes.append((label, float(conf), (x1,y1,x2,y2)))
                            yolo_labels.append((label, float(conf), (x1,y1,x2,y2)))
                            color = (0,255,0) if label.lower()=="person" else (255,0,255) if "phone" in label.lower() else (0,255,255)
                            cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
                            cv2.putText(out,f"{label} {conf:.2f}",(x1,max(y1-5,10)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
                except Exception as e:
                    pass

            # ── Person proximity check ──
            person_close = any(
                (x2-x1)*(y2-y1)/img_area >= min_area_ratio
                for (label,conf,(x1,y1,x2,y2)) in yolo_boxes if label.lower()=="person"
            )
            if not person_close:
                cv2.putText(out,"No person close enough",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
                stframe.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
                if stop_button: break
                continue

            # ── MediaPipe ──
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            face_result = face_landmarker.detect(mp_image) if face_landmarker else None
            hand_results= hand_landmarker.detect(mp_image) if hand_landmarker else None
            pose_results= pose_landmarker.detect(mp_image) if pose_landmarker else None

            face_landmarks = None
            if face_result and getattr(face_result,"face_landmarks",None) and face_result.face_landmarks:
                face_landmarks = LandmarkListAdapter(face_result.face_landmarks[0])

            yaw_rel, pitch_rel, face_ok = 0.0, 0.0, False
            if face_landmarks:
                yaw_rel, pitch_rel, face_ok = estimate_head_angles_solvepnp(face_landmarks, frame.shape)
                for lm in face_landmarks.landmark:
                    cv2.circle(out,(int(lm.x*w),int(lm.y*h)),1,(0,255,0),-1)

            # Calibration triggers
            # ── Auto-calibration: average first 30 good frames as neutral ──
            AUTO_CALIB_FRAMES = 30
            if not st.session_state.get("auto_calib_done", False):
                if face_ok:
                    st.session_state["auto_calib_buffer"].append((yaw_rel, pitch_rel))
                n = len(st.session_state["auto_calib_buffer"])
                if n >= AUTO_CALIB_FRAMES:
                    yaws    = [y for y, p in st.session_state["auto_calib_buffer"]]
                    pitches = [p for y, p in st.session_state["auto_calib_buffer"]]
                    st.session_state["base_yaw"]        = float(np.mean(yaws))
                    st.session_state["base_pitch"]      = float(np.mean(pitches))
                    st.session_state["auto_calib_done"] = True
                else:
                    pct   = int((n / AUTO_CALIB_FRAMES) * 100)
                    bar_w = int((n / AUTO_CALIB_FRAMES) * (w - 40))
                    cv2.rectangle(out, (20, h-50), (w-20, h-30), (20, 28, 40), -1)
                    cv2.rectangle(out, (20, h-50), (20+bar_w, h-30), (0, 200, 100), -1)
                    cv2.putText(out, f"Auto-calibrating... {pct}%  look at screen naturally",
                                (22, h-56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (125, 249, 170), 1)

            # Manual calibration buttons (override)
            if st.session_state.get("calibrate_neutral", False):
                if face_ok:
                    st.session_state["base_yaw"]        = yaw_rel
                    st.session_state["base_pitch"]      = pitch_rel
                    st.session_state["auto_calib_done"] = True
                st.session_state["calibrate_neutral"] = False
            if st.session_state.get("calibrate_away", False):
                if face_ok:
                    st.session_state["away_yaw"]   = yaw_rel
                    st.session_state["away_pitch"]  = pitch_rel
                st.session_state["calibrate_away"] = False

            head_score    = head_direction_score(yaw_rel, pitch_rel, face_ok, head_sensitivity)

            # ── No-face grace: hold last head score for 1.5s before penalising ──
            NO_FACE_HOLD = 1.5
            if not face_ok:
                if st.session_state["no_face_since"] is None:
                    st.session_state["no_face_since"] = now_time
                if now_time - st.session_state["no_face_since"] < NO_FACE_HOLD:
                    head_score = st.session_state.get("last_head_score", 0.5)
            else:
                st.session_state["no_face_since"] = None
                st.session_state["last_head_score"] = head_score
            ear_score     = compute_ear_score(face_landmarks, h, w)
            emotion_score = compute_emotion_score(face_landmarks, h, w)

            pose_landmarks = None
            if pose_results and getattr(pose_results,"pose_landmarks",None) and pose_results.pose_landmarks:
                pose_landmarks = LandmarkListAdapter(pose_results.pose_landmarks[0])
            slouch_score = compute_slouch_score(pose_landmarks, h, w)

            # ── Hands ──
            wrists = []
            mp_hands_detected = False
            task_hands = []
            if hand_results and getattr(hand_results,"hand_landmarks",None):
                task_hands = [LandmarkListAdapter(lms) for lms in hand_results.hand_landmarks]
            if task_hands:
                mp_hands_detected = True
                for hand_lm in task_hands:
                    wx = int(hand_lm.landmark[0].x*w)
                    wy = int(hand_lm.landmark[0].y*h)
                    wrists.append((wx,wy))
                    cv2.circle(out,(wx,wy),5,(255,0,0),-1)
            if not mp_hands_detected and use_yolo:
                for label,conf,(x1,y1,x2,y2) in yolo_boxes:
                    if "hand" in label.lower():
                        cx,cy = (x1+x2)//2,(y1+y2)//2
                        wrists.append((cx,cy))
                        mp_hands_detected = True

            curr_hand_landmarks = task_hands[0] if task_hands else None
            inferred_hand_activity = compute_hand_activity(st.session_state["prev_hand_landmarks"], curr_hand_landmarks)
            real_hand = 1.0 if wrists else 0.0
            st.session_state["prev_hand_landmarks"] = curr_hand_landmarks
            st.session_state["prev_wrists"] = wrists

            book_presence, book_engagement_score = compute_book_engagement(yolo_boxes, wrists)
            if book_presence < 0.5: book_engagement_score = 0.0

            # ── Phone ──
            phone_present_yolo = any("phone" in l.lower() or "mobile" in l.lower() for (l,conf,_) in yolo_labels)
            phone_present = simulate_phone or (phone_present_yolo if use_yolo else False)
            if phone_present:
                st.session_state["phone_timer"] = now_time
            elif now_time - st.session_state["phone_timer"] < 3:
                phone_present = True
            applied_phone_pen = phone_penalty if phone_present else 0.0

            # ── Movement / gaze ──
            person_center = next(((((x1+x2)//2),((y1+y2)//2)) for (label,conf,(x1,y1,x2,y2)) in yolo_boxes if label.lower()=="person"), None)
            movement_score = compute_movement_frequency(st.session_state["prev_person_center"], person_center)
            st.session_state["prev_person_center"] = person_center

            gaze_state = "screen" if pitch_rel > -15 else "book"
            st.session_state["gaze_history"].append(gaze_state)
            if len(st.session_state["gaze_history"]) > 10: st.session_state["gaze_history"].pop(0)
            switch_score = compute_gaze_switch_frequency(st.session_state["gaze_history"])

            # ── Focus score ──
            params = {
                "head_direction": head_score, "inferred_hand_activity": inferred_hand_activity,
                "real_hand": real_hand, "movement_score": movement_score,
                "switch_score": switch_score, "ear_score": ear_score,
                "slouch_score": slouch_score, "emotion_score": emotion_score,
                "book_presence": book_presence, "book_engagement": book_engagement_score,
            }

            def compute_focus(params):
                if user_mode == "Dynamic":
                    w_map = {"movement_score": 0.5, "head_direction": 0.3, "slouch_score": 0.2}
                else:
                    if mode == "Digital":
                        # head is most reliable, ear catches drowsiness, slouch catches posture
                        # emotion removed — consistently broken
                        w_map = {"head_direction": 0.55, "ear_score": 0.25, "slouch_score": 0.20}
                    elif mode == "Offline":
                        w_map = {"book_engagement": 0.5, "inferred_hand_activity": 0.2, "slouch_score": 0.2, "head_direction": 0.1}
                    else:  # Hybrid
                        w_map = {"switch_score": 0.4, "slouch_score": 0.3, "head_direction": 0.3}
                active = {k:v for k,v in w_map.items() if v>0}
                S = sum(active.values())
                if S == 0: return 0.0, w_map
                score = sum(params.get(k,1.0)*active[k] for k in active) / S
                return float(np.clip(score - applied_phone_pen, 0, 1)), w_map

            focus_score, weights_used = compute_focus(params)
            focus_score = min(focus_score + 0.1, 1.0)

            # ── 5-frame rolling average — smooths out single-frame jitter ──
            st.session_state["raw_score_buffer"].append(focus_score)
            if len(st.session_state["raw_score_buffer"]) > 5:
                st.session_state["raw_score_buffer"].pop(0)
            focus_score = float(np.mean(st.session_state["raw_score_buffer"]))

            st.session_state["current_focus_score"] = focus_score

            total_score += focus_score
            frame_count += 1
            st.session_state["session_data"].append(focus_score)

            # ── Rolling score history for graph (keep last 60) ──
            st.session_state["score_history"].append(focus_score)
            if len(st.session_state["score_history"]) > 60:
                st.session_state["score_history"].pop(0)

            # ── Alerts with grace period ──
            # Only fire a distraction after the user has been below threshold
            # continuously for grace_period seconds — avoids false positives.
            if enable_notifications and focus_score < focus_threshold:
                if st.session_state["unfocused_since"] is None:
                    st.session_state["unfocused_since"] = now_time  # start the timer
                unfocused_duration = now_time - st.session_state["unfocused_since"]
                grace = st.session_state["grace_period"]

                if unfocused_duration >= grace:
                    time_since = now_time - st.session_state["last_notification_time"]
                    if time_since >= st.session_state["notification_cooldown"]:
                        st.session_state["distraction_count"] += 1
                        st.session_state["last_notification_time"] = now_time
                        if enable_tts: speak_focus()
                        if ntfy_topic:
                            name_tag  = st.session_state["student_name"] or "Unknown student"
                            threading.Thread(
                                target=send_ntfy,
                                args=(ntfy_topic, f"Focus dropped to {focus_score:.2f} — distracted for {grace:.0f}s", name_tag),
                                daemon=True
                            ).start()
                    # Show toast with a countdown hint while in grace window
                    alert_display.markdown('<div class="focus-toast">⚠ FOCUS — you\'re drifting</div>', unsafe_allow_html=True)
                else:
                    # Still in grace window — show a softer warning
                    remaining = grace - unfocused_duration
                    alert_display.markdown(
                        f'<div style="background:rgba(255,209,102,0.08);border:1px solid #FFD166;border-radius:10px;padding:10px 14px;font-family:\'Space Mono\',monospace;font-size:11px;color:#FFD166;text-align:center">⏱ refocusing… {remaining:.1f}s</div>',
                        unsafe_allow_html=True
                    )
            else:
                # Back above threshold — reset grace timer
                st.session_state["unfocused_since"] = None
                alert_display.empty()

            # ── Update score widget + graph ──
            score_display.markdown(
                render_score_widget(focus_score, st.session_state["distraction_count"]),
                unsafe_allow_html=True
            )
            graph_display.markdown(
                render_score_graph(st.session_state["score_history"]),
                unsafe_allow_html=True
            )

            # ── Camera feed ──
            stframe.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)

            # ── Debug telemetry ──
            _calib_status = "AUTO ✓" if st.session_state.get("auto_calib_done") else f"warming {len(st.session_state.get('auto_calib_buffer',[]))}/30"
            _el  = st.session_state.get("_diag_ear_left",  0.0)
            _er  = st.session_state.get("_diag_ear_right", 0.0)
            _ea  = st.session_state.get("_diag_ear_avg",   0.0)
            _ny  = st.session_state.get("_diag_nose_y",    0.0)
            _shy = st.session_state.get("_diag_shoulder_y",0.0)
            _gap = st.session_state.get("_diag_gap",       0.0)
            debug_placeholder.markdown(f"""
```
━━━ FOCUS  {focus_score:.3f}  ━━━  Calibration: {_calib_status}
           Base yaw:{st.session_state['base_yaw']:.1f}°  pitch:{st.session_state['base_pitch']:.1f}°

HEAD       score={head_score:.3f}   yaw={yaw_rel:.1f}°  pitch={pitch_rel:.1f}°

EAR        score={ear_score:.3f}
           left={_el:.4f}  right={_er:.4f}  avg={_ea:.4f}
           thresholds: open={0.30}  closed={0.08}
           → open eye ~0.36, closed eye ~0.035

SLOUCH     score={slouch_score:.3f}
           nose_y={_ny:.4f}  shoulder_y={_shy:.4f}  gap={_gap:.4f}
           thresholds: ideal_nose={0.42}  bad_nose={0.55}
           → straight ~0.37, slouch ~0.48

EMOTION    {emotion_score:.3f}
HAND real  {real_hand:.1f}    inferred={inferred_hand_activity:.3f}
BOOK       presence={book_presence:.1f}  engagement={book_engagement_score:.3f}
MOVEMENT   {movement_score:.3f}   switch={switch_score:.3f}
PHONE      {'YES ⚠' if phone_present else 'no'}   penalty={applied_phone_pen:.2f}
BLINKS     {st.session_state['blink_counter']} / {st.session_state['total_frames']} frames

WEIGHTS    {weights_used}
```""", unsafe_allow_html=False)

            if stop_button:
                break

        cap.release()
        stframe.empty()
        st.session_state["session_running"] = False

        # ── Session summary ──
        if frame_count > 0:
            avg_score        = total_score / frame_count
            session_duration = time.time() - st.session_state["session_start"]
            summary_placeholder.markdown(
                render_session_summary(avg_score, frame_count, st.session_state["distraction_count"], session_duration),
                unsafe_allow_html=True
            )
