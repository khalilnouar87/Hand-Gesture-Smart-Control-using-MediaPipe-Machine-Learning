import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .card {
        background: linear-gradient(135deg, #1e2130, #2a2d3e);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #3a3f5c;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
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
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #3a3f5c, transparent);
        margin: 16px 0;
    }
    .result-box {
        background: linear-gradient(135deg, #1e2130, #2a2d3e);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        border: 2px solid #7c4dff;
        margin: 10px 0;
    }
    .result-icon { font-size: 80px; margin-bottom: 10px; }
    .result-name { font-size: 32px; font-weight: bold; color: #7c4dff; }
    .result-conf { font-size: 16px; color: #90a4ae; margin-top: 8px; }
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
CONTROL_MAP = {
    0: ("Move Mouse",  "🖱️", "#7c4dff"),
    1: ("Stop Mouse",  "✋",  "#546e7a"),
    2: ("Left Click",  "👆", "#00bcd4"),
    3: ("Right Click", "☝️", "#ff9800"),
    4: ("Scroll",      "📜", "#4caf50"),
    5: ("Volume Up",   "🔊", "#00c853"),
    6: ("Volume Down", "🔉", "#f44336"),
    7: ("Screenshot",  "📸", "#e91e63"),
}

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ─────────────────────────────────────────────
#  Load Model & MediaPipe
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("model.p", "rb") as f:
            d = pickle.load(f)
        return d["model"], True
    except Exception:
        return None, False

@st.cache_resource
def get_mediapipe():
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_hands, mp_drawing, mp_styles, hands

# ─────────────────────────────────────────────
#  Helpers
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

def hex_to_bgr(h):
    r = int(h[1:3], 16)
    g = int(h[3:5], 16)
    b = int(h[5:7], 16)
    return (b, g, r)

def draw_overlay(frame, hand_lm, label, color_hex,
                 H, W, mp_drawing, mp_hands, mp_styles):
    mp_drawing.draw_landmarks(
        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
        mp_styles.get_default_hand_landmarks_style(),
        mp_styles.get_default_hand_connections_style()
    )
    xs = [lm.x for lm in hand_lm.landmark]
    ys = [lm.y for lm in hand_lm.landmark]
    x1 = max(int(min(xs) * W) - 10, 0)
    y1 = max(int(min(ys) * H) - 10, 0)
    x2 = min(int(max(xs) * W) + 10, W)
    y2 = min(int(max(ys) * H) + 10, H)
    bgr = hex_to_bgr(color_hex)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame,
                  (x1, max(y1 - th - 14, 0)),
                  (x1 + tw + 10, y1), bgr, -1)
    cv2.putText(frame, label, (x1 + 5, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return frame

# ─────────────────────────────────────────────
#  Video Processor
# ─────────────────────────────────────────────
class GestureProcessor(VideoProcessorBase):

    def __init__(self):
        self.model      = None
        self.mp_hands   = None
        self.mp_drawing = None
        self.mp_styles  = None
        self.hands      = None

        # shared result
        self.current_gesture = "None"
        self.current_icon    = "🖐️"
        self.current_color   = "#7c4dff"
        self.confidence      = 0.0
        self.frame_count     = 0

        self._load()

    def _load(self):
        model, loaded = load_model()
        if loaded:
            self.model = model

        mp_h, mp_d, mp_s, hands = get_mediapipe()
        self.mp_hands   = mp_h
        self.mp_drawing = mp_d
        self.mp_styles  = mp_s
        self.hands      = hands

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        H, W, _ = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        self.frame_count += 1

        if results.multi_hand_landmarks and self.model is not None:
            for hand_lm in results.multi_hand_landmarks:
                norm_data = normalize_landmarks(hand_lm)
                pred      = self.model.predict([norm_data])[0]
                proba     = self.model.predict_proba([norm_data])[0]
                class_id  = int(pred)
                conf      = float(np.max(proba)) * 100

                name, icon, color = CONTROL_MAP.get(
                    class_id, ("Unknown", "❓", "#fff"))

                # Update shared state
                self.current_gesture = name
                self.current_icon    = icon
                self.current_color   = color
                self.confidence      = conf

                # Draw on frame
                img = draw_overlay(
                    img, hand_lm, f"{name} {conf:.0f}%",
                    color, H, W,
                    self.mp_drawing,
                    self.mp_hands,
                    self.mp_styles
                )

            # Frame info
            cv2.putText(img,
                        f"Frame: {self.frame_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 200), 2)
            cv2.putText(img,
                        f"Conf: {self.confidence:.1f}%",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (100, 220, 255), 2)
        else:
            self.current_gesture = "None"
            self.current_icon    = "🖐️"
            self.current_color   = "#7c4dff"
            self.confidence      = 0.0

            cv2.putText(img,
                        "No Hand Detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (100, 100, 100), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─────────────────────────────────────────────
#  Session State
# ─────────────────────────────────────────────
defaults = {
    "model_loaded": False,
    "history":      [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### 📦 Model")
    model, loaded = load_model()
    if loaded:
        st.session_state.model_loaded = True
        st.success("✅ Model loaded!")
    else:
        st.warning("⚠️ model.p not found")
        uploaded_model = st.file_uploader("Upload model.p", type=["p"])
        if uploaded_model:
            with open("model.p", "wb") as f:
                f.write(uploaded_model.read())
            st.success("✅ Uploaded! Rerunning...")
            st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### 🗂️ Gesture Map")
    for cid, (name, icon, color) in CONTROL_MAP.items():
        st.markdown(
            f'<div class="gesture-item">'
            f'<span style="font-size:20px;margin-right:10px;">{icon}</span>'
            f'<span><b style="color:{color};">{cid}</b> — {name}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown(
    '<div class="title-text">🖐️ Hand Gesture Recognition</div>',
    unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">'
    'Live webcam · MediaPipe + Random Forest'
    '</div>',
    unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────────
col_cam, col_info = st.columns([3, 1])

with col_cam:
    st.markdown("### 📹 Live Camera")

    if not st.session_state.model_loaded:
        st.error("⚠️ Upload model.p in the sidebar first!")
    else:
        ctx = webrtc_streamer(
            key="gesture",
            video_processor_factory=GestureProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True,
        )

with col_info:
    st.markdown("### 🎯 Live Result")
    result_ph = st.empty()
    st.markdown("### 📊 Confidence")
    conf_ph   = st.empty()
    st.markdown("### 🕓 History")
    hist_ph   = st.empty()

# ─────────────────────────────────────────────
#  Live Result Update Loop
# ─────────────────────────────────────────────
if st.session_state.model_loaded:
    if "ctx" in dir() and ctx.video_processor:
        while True:
            processor = ctx.video_processor

            gesture = processor.current_gesture
            icon    = processor.current_icon
            color   = processor.current_color
            conf    = processor.confidence

            # Update result box
            result_ph.markdown(
                f'<div class="result-box" style="border-color:{color};">'
                f'<div class="result-icon">{icon}</div>'
                f'<div class="result-name" style="color:{color};">{gesture}</div>'
                f'<div class="result-conf">{conf:.1f}% confidence</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Confidence bar
            conf_ph.markdown(
                f'<div class="card">'
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:14px;color:#e0e0e0;">'
                f'<span>{icon} {gesture}</span>'
                f'<span style="color:{color};">{conf:.1f}%</span></div>'
                f'<div style="background:#0f1117;border-radius:6px;'
                f'height:10px;margin-top:8px;">'
                f'<div style="width:{conf}%;background:{color};'
                f'height:10px;border-radius:6px;'
                f'transition:width 0.3s;"></div></div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Save to history
            if gesture != "None":
                if (not st.session_state.history or
                        st.session_state.history[-1]["name"] != gesture):
                    st.session_state.history.append({
                        "name": gesture,
                        "icon": icon,
                        "conf": conf,
                        "time": time.strftime("%H:%M:%S")
                    })
                    if len(st.session_state.history) > 10:
                        st.session_state.history.pop(0)

            # History list
            hist_html = '<div class="card" style="max-height:300px;overflow-y:auto;">'
            for entry in reversed(st.session_state.history):
                c = "#7c4dff"
                for _, (n, _, cl) in CONTROL_MAP.items():
                    if n == entry["name"]:
                        c = cl
                        break
                hist_html += (
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:6px 0;border-bottom:1px solid #3a3f5c;">'
                    f'<span>{entry["icon"]} '
                    f'<span style="color:{c};">{entry["name"]}</span></span>'
                    f'<span style="color:#546e7a;font-size:12px;">'
                    f'{entry["conf"]:.0f}% · {entry["time"]}</span>'
                    f'</div>'
                )
            hist_html += '</div>'
            hist_ph.markdown(hist_html, unsafe_allow_html=True)

            time.sleep(0.1)
