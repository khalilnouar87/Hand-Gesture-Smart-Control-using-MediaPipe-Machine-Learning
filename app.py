# app.py - Streamlit Cloud Compatible (No Camera, No pyautogui)
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from PIL import Image

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
    .result-icon  { font-size: 80px; margin-bottom: 10px; }
    .result-name  { font-size: 32px; font-weight: bold; color: #7c4dff; }
    .result-conf  { font-size: 16px; color: #90a4ae; margin-top: 8px; }
    .prob-row {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        color: #e0e0e0;
        margin: 5px 0;
    }
    .prob-bar-bg {
        background: #0f1117;
        border-radius: 4px;
        height: 6px;
        margin-top: 3px;
    }
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

# ─────────────────────────────────────────────
#  Session State
# ─────────────────────────────────────────────
defaults = {
    "model":          None,
    "model_loaded":   False,
    "history":        [],
    "total_tested":   0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
#  Load Model
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
        static_image_mode=True,
        min_detection_confidence=0.3
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


def draw_overlay(frame_rgb, hand_lm, label, color_hex,
                 H, W, mp_drawing, mp_hands, mp_styles):
    mp_drawing.draw_landmarks(
        frame_rgb, hand_lm, mp_hands.HAND_CONNECTIONS,
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
    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), bgr, 2)

    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(
        frame_rgb,
        (x1, max(y1 - th - 14, 0)),
        (x1 + tw + 10, y1),
        bgr, -1
    )
    cv2.putText(
        frame_rgb, label, (x1 + 5, max(y1 - 6, 0)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (255, 255, 255), 2, cv2.LINE_AA
    )
    return frame_rgb

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Model loader
    st.markdown("### 📦 Model")
    model, loaded = load_model()
    if loaded:
        st.session_state.model        = model
        st.session_state.model_loaded = True
        st.success("✅ Model loaded!")
    else:
        st.warning("⚠️ model.p not found in repo")
        uploaded_model = st.file_uploader(
            "Upload model.p", type=["p"])
        if uploaded_model:
            with open("model.p", "wb") as f:
                f.write(uploaded_model.read())
            st.success("✅ Saved! Click Rerun.")
            st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Info
    st.info(
        "🌐 **Streamlit Cloud Mode**\n\n"
        "Upload a hand photo to detect the gesture.\n\n"
        "Mouse & keyboard control requires running **locally**."
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Gesture Map
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

    # Clear history
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history     = []
        st.session_state.total_tested = 0
        st.rerun()

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown(
    '<div class="title-text">🖐️ Hand Gesture Recognition</div>',
    unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">'
    'Upload a hand image · MediaPipe + Random Forest'
    '</div>',
    unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.markdown(
    f'<div class="metric-box">'
    f'<div class="metric-value">{st.session_state.total_tested}</div>'
    f'<div class="metric-label">Images Tested</div></div>',
    unsafe_allow_html=True)
c2.markdown(
    f'<div class="metric-box">'
    f'<div class="metric-value">{len(st.session_state.history)}</div>'
    f'<div class="metric-label">Detections</div></div>',
    unsafe_allow_html=True)

last_gesture = st.session_state.history[-1] if st.session_state.history else None
c3.markdown(
    f'<div class="metric-box">'
    f'<div class="metric-value" style="font-size:20px;">'
    f'{last_gesture["icon"] + " " + last_gesture["name"] if last_gesture else "—"}'
    f'</div>'
    f'<div class="metric-label">Last Gesture</div></div>',
    unsafe_allow_html=True)
c4.markdown(
    f'<div class="metric-box">'
    f'<div class="metric-value">'
    f'{f"{last_gesture[chr(99)+chr(111)+chr(110)+chr(102)]:.1f}%" if last_gesture else "—"}'
    f'</div>'
    f'<div class="metric-label">Last Confidence</div></div>',
    unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  UPLOAD & PREDICT
# ─────────────────────────────────────────────
st.markdown("### 📤 Upload Hand Image")

uploaded_imgs = st.file_uploader(
    "Choose one or more hand images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_imgs and st.session_state.model_loaded:
    mp_hands_mod, mp_drawing_mod, mp_styles_mod, hands_det = get_mediapipe()
    model_clf = st.session_state.model

    for uploaded_img in uploaded_imgs:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        file_bytes = np.asarray(
            bytearray(uploaded_img.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W, _ = img_rgb.shape

        st.session_state.total_tested += 1

        results = hands_det.process(img_rgb)

        col_img, col_result = st.columns([3, 2])

        if results.multi_hand_landmarks:
            all_preds = []

            for hand_lm in results.multi_hand_landmarks:
                norm_data = normalize_landmarks(hand_lm)
                pred      = model_clf.predict([norm_data])[0]
                proba     = model_clf.predict_proba([norm_data])[0]
                class_id  = int(pred)
                conf      = float(np.max(proba)) * 100

                name, icon, color = CONTROL_MAP.get(
                    class_id, ("Unknown", "❓", "#fff"))

                img_rgb = draw_overlay(
                    img_rgb, hand_lm, name, color,
                    H, W, mp_drawing_mod,
                    mp_hands_mod, mp_styles_mod
                )
                all_preds.append((class_id, name, icon, color, conf, proba))

                # Save to history
                st.session_state.history.append({
                    "name": name,
                    "icon": icon,
                    "conf": conf,
                    "time": time.strftime("%H:%M:%S")
                })

            with col_img:
                st.image(
                    img_rgb,
                    caption=f"📁 {uploaded_img.name}",
                    use_container_width=True
                )

            with col_result:
                for class_id, name, icon, color, conf, proba in all_preds:
                    # Result box
                    st.markdown(
                        f'<div class="result-box" style="border-color:{color};">'
                        f'<div class="result-icon">{icon}</div>'
                        f'<div class="result-name" style="color:{color};">{name}</div>'
                        f'<div class="result-conf">Confidence: {conf:.1f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    # Metrics row
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Class ID",    class_id)
                    m2.metric("Confidence",  f"{conf:.1f}%")
                    m3.metric("Hands Found", len(results.multi_hand_landmarks))

                    # Probability bars for all classes
                    st.markdown("#### 📊 All Class Probabilities")
                    classes = model_clf.classes_
                    for i, cls in enumerate(classes):
                        cid      = int(cls)
                        cname, cicon, ccolor = CONTROL_MAP.get(
                            cid, ("Unknown", "❓", "#fff"))
                        pct = proba[i] * 100
                        st.markdown(
                            f'<div class="prob-row">'
                            f'<span>{cicon} {cname}</span>'
                            f'<span style="color:{ccolor};">{pct:.1f}%</span>'
                            f'</div>'
                            f'<div class="prob-bar-bg">'
                            f'<div style="width:{pct}%;background:{ccolor};'
                            f'height:6px;border-radius:4px;"></div></div>',
                            unsafe_allow_html=True
                        )

        else:
            with col_img:
                st.image(
                    img_rgb,
                    caption=f"📁 {uploaded_img.name}",
                    use_container_width=True
                )
            with col_result:
                st.markdown(
                    '<div class="result-box" style="border-color:#f44336;">'
                    '<div class="result-icon">❌</div>'
                    '<div class="result-name" style="color:#f44336;">No Hand Detected</div>'
                    '<div class="result-conf">Try a clearer image with good lighting</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.markdown("#### 💡 Tips")
                st.markdown("""
                - ✅ Make sure your **hand is clearly visible**
                - ✅ Use **good lighting**
                - ✅ Keep hand **centered** in frame
                - ✅ Avoid **cluttered backgrounds**
                - ✅ Try a **closer** shot
                """)

elif uploaded_imgs and not st.session_state.model_loaded:
    st.error("⚠️ Please upload model.p in the sidebar first!")

else:
    # Empty state
    st.markdown("""
        <div class="card" style="text-align:center; padding: 60px 20px;">
            <div style="font-size:80px;">🖐️</div>
            <div style="font-size:24px;font-weight:bold;
                        color:#7c4dff;margin-top:16px;">
                Upload a Hand Image to Start
            </div>
            <div style="color:#546e7a;margin-top:12px;font-size:15px;">
                Supports JPG, JPEG, PNG · Multiple images supported
            </div>
        </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HISTORY
# ─────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🕓 Detection History")

    cols = st.columns(4)
    for i, entry in enumerate(reversed(st.session_state.history[-12:])):
        with cols[i % 4]:
            name, icon = entry["name"], entry["icon"]
            conf       = entry["conf"]
            ts         = entry["time"]
            color      = "#7c4dff"
            for _, (n, _, c) in CONTROL_MAP.items():
                if n == name:
                    color = c
                    break
            st.markdown(
                f'<div class="card" style="text-align:center;padding:14px;">'
                f'<div style="font-size:36px;">{icon}</div>'
                f'<div style="font-weight:bold;color:{color};font-size:14px;">{name}</div>'
                f'<div style="color:#90a4ae;font-size:12px;">{conf:.1f}% · {ts}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
