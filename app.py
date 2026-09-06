# app.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle

import time
from PIL import Image
import threading

# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Hand Gesture Control",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0f1117; }

    /* Card style */
    .card {
        background: linear-gradient(135deg, #1e2130, #2a2d3e);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #3a3f5c;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    /* Status badge */
    .badge-active {
        background: linear-gradient(90deg, #00c853, #69f0ae);
        color: #000;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    .badge-idle {
        background: linear-gradient(90deg, #546e7a, #90a4ae);
        color: #fff;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }

    /* Gesture card */
    .gesture-item {
        display: flex;
        align-items: center;
        background: #1a1d2e;
        border-radius: 10px;
        padding: 10px 16px;
        margin: 6px 0;
        border-left: 4px solid #7c4dff;
        font-size: 15px;
        color: #e0e0e0;
    }

    /* Metric box */
    .metric-box {
        background: #1a1d2e;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #3a3f5c;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #7c4dff;
    }
    .metric-label {
        font-size: 13px;
        color: #90a4ae;
        margin-top: 4px;
    }

    /* Action log */
    .log-entry {
        background: #1a1d2e;
        border-radius: 8px;
        padding: 8px 14px;
        margin: 4px 0;
        font-size: 13px;
        color: #b0bec5;
        border-left: 3px solid #00bcd4;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar */
    .css-1d391kg { background-color: #1a1d2e; }

    /* Title */
    .title-text {
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(90deg, #7c4dff, #00bcd4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 4px;
    }
    .subtitle-text {
        text-align: center;
        color: #90a4ae;
        font-size: 15px;
        margin-bottom: 24px;
    }

    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #3a3f5c, transparent);
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
MODEL_PATH = "model.p"

CONTROL_MAP = {
    0: ("Move Mouse",    "🖱️",  "#7c4dff"),
    1: ("Stop Mouse",    "✋",  "#546e7a"),
    2: ("Left Click",    "👆",  "#00bcd4"),
    3: ("Right Click",   "☝️",  "#ff9800"),
    4: ("Scroll",        "📜",  "#4caf50"),
    5: ("Volume Up",     "🔊",  "#00c853"),
    6: ("Volume Down",   "🔉",  "#f44336"),
    7: ("Screenshot",    "📸",  "#e91e63"),
}

# ─────────────────────────────────────────────
#  Session State Init
# ─────────────────────────────────────────────
defaults = {
    "running":          False,
    "action_log":       [],
    "current_gesture":  "None",
    "gesture_counts":   {v[0]: 0 for v in CONTROL_MAP.values()},
    "total_frames":     0,
    "detections":       0,
    "model_loaded":     False,
    "model":            None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
#  Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    try:
        with open(path, "rb") as f:
            d = pickle.load(f)
        return d["model"], True
    except FileNotFoundError:
        return None, False
    except Exception as e:
        return None, False

# ─────────────────────────────────────────────
#  MediaPipe Setup
# ─────────────────────────────────────────────
@st.cache_resource
def get_mediapipe():
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return mp_hands, mp_drawing, mp_styles, hands

# ─────────────────────────────────────────────
#  Helper Functions
# ─────────────────────────────────────────────
def normalize_landmarks(hand_landmarks):
    x_ = [lm.x for lm in hand_landmarks.landmark]
    y_ = [lm.y for lm in hand_landmarks.landmark]
    min_x, min_y = min(x_), min(y_)
    data_aux = []
    for lm in hand_landmarks.landmark:
        data_aux.append(lm.x - min_x)
        data_aux.append(lm.y - min_y)
    return np.asarray(data_aux)


def add_log(action_name, icon):
    ts = time.strftime("%H:%M:%S")
    entry = f"{ts}  {icon}  {action_name}"
    st.session_state.action_log.insert(0, entry)
    # keep last 20
    if len(st.session_state.action_log) > 20:
        st.session_state.action_log.pop()


last_action_time = [time.time()]
ACTION_COOLDOWN  = 0.5
SMOOTHNESS       = 0.5


def execute_action(class_id, hand_landmarks, mp_hands_ref):
    now = time.time()
    name, icon, _ = CONTROL_MAP.get(class_id, ("Unknown", "❓", "#fff"))

    if class_id == 0:           # Move mouse
        tip = hand_landmarks.landmark[mp_hands_ref.HandLandmark.INDEX_FINGER_TIP]
        sw, sh = pyautogui.size()
        tx, ty = tip.x * sw, tip.y * sh
        mx, my = pyautogui.position()
        nx = int(mx + (tx - mx) * SMOOTHNESS)
        ny = int(my + (ty - my) * SMOOTHNESS)
        pyautogui.moveTo(nx, ny, _pause=False)

    elif class_id == 1:         # Stop – do nothing
        pass

    elif now - last_action_time[0] >= ACTION_COOLDOWN:
        if   class_id == 2: pyautogui.click()
        elif class_id == 3: pyautogui.rightClick()
        elif class_id == 4: pyautogui.scroll(-10)
        elif class_id == 5: pyautogui.press("volumeup")
        elif class_id == 6: pyautogui.press("volumedown")
        elif class_id == 7:
            fname = f"screenshot_{int(time.time())}.png"
            pyautogui.screenshot(fname)
        last_action_time[0] = now
        add_log(name, icon)

    return name, icon


def draw_overlay(frame, hand_landmarks, label, color_hex, H, W, mp_drawing, mp_hands, mp_styles):
    # Landmarks
    mp_drawing.draw_landmarks(
        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_styles.get_default_hand_landmarks_style(),
        mp_styles.get_default_hand_connections_style()
    )
    # Bounding box
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    x1, y1 = int(min(xs)*W)-10, int(min(ys)*H)-10
    x2, y2 = int(max(xs)*W)+10, int(max(ys)*H)+10

    # Convert hex to BGR
    r = int(color_hex[1:3], 16)
    g = int(color_hex[3:5], 16)
    b = int(color_hex[5:7], 16)
    bgr = (b, g, r)

    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)

    # Label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x1, y1-th-14), (x1+tw+10, y1), bgr, -1)
    cv2.putText(frame, label, (x1+5, y1-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    return frame

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Model loader
    st.markdown("### 📦 Model")
    model, loaded = load_model(MODEL_PATH)
    if loaded:
        st.session_state.model = model
        st.session_state.model_loaded = True
        st.success("✅ Model loaded!")
    else:
        st.error("❌ model.p not found")
        uploaded = st.file_uploader("Upload model.p", type=["p"])
        if uploaded:
            with open("model.p", "wb") as f:
                f.write(uploaded.read())
            st.success("Uploaded! Reload the page.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Camera settings
    st.markdown("### 📷 Camera")
    cam_index   = st.selectbox("Camera Index", [0, 1, 2], index=0)
    frame_width = st.slider("Frame Width",  320, 1280, 640, step=160)
    frame_height= st.slider("Frame Height", 240, 720,  480, step=120)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Controls settings
    st.markdown("### 🎮 Controls")
    smoothness_val = st.slider("Mouse Smoothness", 0.1, 1.0, 0.5, 0.05)
    SMOOTHNESS     = smoothness_val
    cooldown_val   = st.slider("Action Cooldown (s)", 0.1, 2.0, 0.5, 0.1)
    ACTION_COOLDOWN= cooldown_val

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Gesture legend
    st.markdown("### 🗂️ Gesture Map")
    for cid, (name, icon, color) in CONTROL_MAP.items():
        st.markdown(
            f'<div class="gesture-item">'
            f'<span style="font-size:20px;margin-right:10px;">{icon}</span>'
            f'<span><b style="color:{color};">{cid}</b> — {name}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────────
st.markdown('<div class="title-text">🖐️ Hand Gesture Smart Control</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Real-time hand gesture recognition powered by MediaPipe & Random Forest</div>', unsafe_allow_html=True)

# ── Top metrics row ──────────────────────────
m1, m2, m3, m4 = st.columns(4)

total_frames_ph   = m1.empty()
detections_ph     = m2.empty()
current_gest_ph   = m3.empty()
status_ph         = m4.empty()

def render_metrics():
    total_frames_ph.markdown(
        f'<div class="metric-box"><div class="metric-value">{st.session_state.total_frames}</div>'
        f'<div class="metric-label">Frames Processed</div></div>', unsafe_allow_html=True)
    detections_ph.markdown(
        f'<div class="metric-box"><div class="metric-value">{st.session_state.detections}</div>'
        f'<div class="metric-label">Hand Detections</div></div>', unsafe_allow_html=True)
    current_gest_ph.markdown(
        f'<div class="metric-box"><div class="metric-value" style="font-size:22px;">'
        f'{st.session_state.current_gesture}</div>'
        f'<div class="metric-label">Current Gesture</div></div>', unsafe_allow_html=True)
    badge = '<span class="badge-active">● RUNNING</span>' if st.session_state.running \
            else '<span class="badge-idle">○ STOPPED</span>'
    status_ph.markdown(
        f'<div class="metric-box" style="padding-top:22px;">{badge}</div>',
        unsafe_allow_html=True)

render_metrics()

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Main content: video | log ────────────────
col_video, col_log = st.columns([3, 1])

with col_video:
    st.markdown("### 📹 Live Camera Feed")
    video_placeholder = st.empty()

with col_log:
    st.markdown("### 📋 Action Log")
    log_placeholder = st.empty()

    st.markdown("### 📊 Gesture Stats")
    stats_placeholder = st.empty()

# ── Control buttons ───────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])

start_btn = btn_col1.button("▶ Start", type="primary",  use_container_width=True)
stop_btn  = btn_col2.button("⏹ Stop",  type="secondary", use_container_width=True)
clear_btn = btn_col3.button("🗑️ Clear Log & Stats",      use_container_width=True)

if start_btn and st.session_state.model_loaded:
    st.session_state.running = True

if stop_btn:
    st.session_state.running = False

if clear_btn:
    st.session_state.action_log     = []
    st.session_state.gesture_counts = {v[0]: 0 for v in CONTROL_MAP.values()}
    st.session_state.total_frames   = 0
    st.session_state.detections     = 0

if start_btn and not st.session_state.model_loaded:
    st.error("⚠️ Please load the model first (upload in sidebar).")

# ─────────────────────────────────────────────
#  Log & Stats render helpers
# ─────────────────────────────────────────────
def render_log():
    if not st.session_state.action_log:
        log_placeholder.markdown(
            '<div class="card" style="color:#546e7a;text-align:center;">No actions yet…</div>',
            unsafe_allow_html=True)
    else:
        html = '<div class="card" style="max-height:420px;overflow-y:auto;">'
        for entry in st.session_state.action_log:
            html += f'<div class="log-entry">{entry}</div>'
        html += '</div>'
        log_placeholder.markdown(html, unsafe_allow_html=True)


def render_stats():
    html = '<div class="card">'
    for name, count in st.session_state.gesture_counts.items():
        pct = 0
        if st.session_state.detections > 0:
            pct = int(count / st.session_state.detections * 100)
        # find color
        color = "#7c4dff"
        for _, (n, _, c) in CONTROL_MAP.items():
            if n == name:
                color = c
                break
        html += f"""
        <div style="margin:8px 0;">
          <div style="display:flex;justify-content:space-between;font-size:13px;color:#e0e0e0;">
            <span>{name}</span><span style="color:{color};">{count}</span>
          </div>
          <div style="background:#0f1117;border-radius:6px;height:6px;margin-top:4px;">
            <div style="width:{pct}%;background:{color};height:6px;border-radius:6px;"></div>
          </div>
        </div>"""
    html += '</div>'
    stats_placeholder.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
if st.session_state.running and st.session_state.model_loaded:

    mp_hands_mod, mp_drawing_mod, mp_styles_mod, hands_detector = get_mediapipe()
    model_clf = st.session_state.model

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    if not cap.isOpened():
        st.error("❌ Cannot open camera!")
        st.session_state.running = False
    else:
        st.toast("🎬 Camera started!", icon="✅")

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Frame read failed, retrying…")
                continue

            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands_detector.process(frame_rgb)
            st.session_state.total_frames += 1

            detected_gesture = "None"

            if results.multi_hand_landmarks:
                st.session_state.detections += 1
                for hand_lm in results.multi_hand_landmarks:
                    norm_data  = normalize_landmarks(hand_lm)
                    pred       = model_clf.predict([norm_data])[0]
                    class_id   = int(pred)

                    name, icon, color = CONTROL_MAP.get(class_id, ("Unknown", "❓", "#fff"))
                    detected_gesture  = f"{icon} {name}"

                    # execute
                    execute_action(class_id, hand_lm, mp_hands_mod)

                    # update stats
                    st.session_state.gesture_counts[name] = \
                        st.session_state.gesture_counts.get(name, 0) + 1

                    # draw on frame
                    frame_rgb = draw_overlay(
                        frame_rgb, hand_lm, name, color,
                        H, W, mp_drawing_mod, mp_hands_mod, mp_styles_mod
                    )

            st.session_state.current_gesture = detected_gesture

            # ── FPS overlay ──────────────────
            cv2.putText(
                frame_rgb,
                f"Frames: {st.session_state.total_frames}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2
            )
            cv2.putText(
                frame_rgb,
                f"Gesture: {detected_gesture.split(' ')[-1] if detected_gesture != 'None' else 'None'}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 220, 255), 2
            )

            # ── Render ───────────────────────
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            render_metrics()
            render_log()
            render_stats()

        cap.release()
        st.toast("⏹ Camera stopped.", icon="ℹ️")

else:
    # Placeholder when not running
    video_placeholder.markdown(
        """
        <div class="card" style="height:420px;display:flex;flex-direction:column;
             align-items:center;justify-content:center;text-align:center;">
            <div style="font-size:80px;">🖐️</div>
            <div style="font-size:22px;font-weight:bold;color:#7c4dff;margin-top:16px;">
                Ready to Start
            </div>
            <div style="color:#546e7a;margin-top:8px;">
                Press <b>▶ Start</b> to activate the camera
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    render_log()
    render_stats()
