```python
import os
import time
import pickle
import threading

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Gesture AI",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(124, 77, 255, 0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 85% 20%,
                rgba(0, 188, 212, 0.10),
                transparent 30%
            ),
            #080a10;
        color: #f5f7fa;
    }

    .main {
        background: transparent;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* ---------- TITLE ---------- */

    .hero {
        padding: 12px 0 25px 0;
        text-align: center;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 900;
        letter-spacing: -1.5px;

        background: linear-gradient(
            90deg,
            #8b5cf6,
            #06b6d4,
            #8b5cf6
        );

        background-size: 200% auto;

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;

        animation: gradient 5s linear infinite;
    }

    @keyframes gradient {
        to {
            background-position: 200% center;
        }
    }

    .hero-subtitle {
        margin-top: 7px;
        color: #8992a3;
        font-size: 15px;
    }


    /* ---------- CARDS ---------- */

    .glass-card {
        background: rgba(19, 23, 34, 0.78);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 20px;

        box-shadow:
            0 15px 40px rgba(0,0,0,0.28),
            inset 0 1px 0 rgba(255,255,255,0.03);

        backdrop-filter: blur(14px);
    }


    /* ---------- RESULT ---------- */

    .result-card {
        background:
            linear-gradient(
                145deg,
                rgba(124,77,255,0.14),
                rgba(6,182,212,0.07)
            );

        border: 1px solid rgba(124,77,255,0.35);

        border-radius: 20px;

        padding: 28px 18px;

        text-align: center;

        min-height: 220px;

        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;

        box-shadow:
            0 15px 45px rgba(0,0,0,0.30);
    }

    .result-icon {
        font-size: 64px;
        line-height: 1;
        margin-bottom: 15px;
    }

    .result-name {
        font-size: 26px;
        font-weight: 800;
    }

    .result-confidence {
        color: #8b95a7;
        margin-top: 8px;
        font-size: 14px;
    }


    /* ---------- METRICS ---------- */

    .metric-card {
        background: rgba(19, 23, 34, 0.78);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 15px;
        padding: 15px;
        text-align: center;
    }

    .metric-number {
        font-size: 24px;
        font-weight: 800;
        color: #8b5cf6;
    }

    .metric-label {
        color: #778196;
        font-size: 12px;
        margin-top: 3px;
    }


    /* ---------- GESTURE ITEMS ---------- */

    .gesture-item {
        display: flex;
        align-items: center;

        background: rgba(14, 17, 25, 0.8);

        border-radius: 11px;

        padding: 10px 12px;

        margin-bottom: 7px;

        border: 1px solid rgba(255,255,255,0.045);
    }

    .gesture-icon {
        font-size: 21px;
        margin-right: 10px;
    }

    .gesture-name {
        color: #e7eaf0;
        font-size: 13px;
    }


    /* ---------- STATUS ---------- */

    .status {
        display: inline-flex;
        align-items: center;
        gap: 8px;

        padding: 7px 12px;

        border-radius: 50px;

        background: rgba(34,197,94,0.10);

        border: 1px solid rgba(34,197,94,0.25);

        color: #4ade80;

        font-size: 12px;
        font-weight: 600;
    }

    .status-dot {
        width: 7px;
        height: 7px;

        border-radius: 50%;

        background: #4ade80;

        box-shadow: 0 0 10px #4ade80;
    }


    /* ---------- HISTORY ---------- */

    .history-row {
        display: flex;
        justify-content: space-between;
        align-items: center;

        padding: 9px 4px;

        border-bottom: 1px solid rgba(255,255,255,0.05);

        font-size: 13px;
    }

    .history-time {
        color: #667085;
        font-size: 11px;
    }


    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #0b0e15;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    section[data-testid="stSidebar"] h2 {
        color: #f5f7fa;
    }


    /* ---------- BUTTON ---------- */

    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
        background: #151925;
        color: #e8ebf0;
    }

    .stButton > button:hover {
        border-color: #7c4dff;
        color: #ffffff;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

CONTROL_MAP = {
    0: ("Move Mouse", "🖱️", "#8b5cf6"),
    1: ("Stop Mouse", "✋", "#64748b"),
    2: ("Left Click", "👆", "#06b6d4"),
    3: ("Right Click", "☝️", "#f59e0b"),
    4: ("Scroll", "📜", "#22c55e"),
    5: ("Volume Up", "🔊", "#10b981"),
    6: ("Volume Down", "🔉", "#ef4444"),
    7: ("Screenshot", "📸", "#ec4899"),
}


RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": ["stun:stun.l.google.com:19302"]
            }
        ]
    }
)


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model.p",
)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    try:

        with open(MODEL_PATH, "rb") as file:
            data = pickle.load(file)

        # Your original model.p structure:
        # {"model": trained_model}

        if isinstance(data, dict) and "model" in data:
            return data["model"]

        # Also support directly saved models
        return data

    except Exception as error:

        st.error(
            f"Could not load model.p: {error}"
        )

        return None


# ============================================================
# LANDMARK NORMALIZATION
# ============================================================

def normalize_landmarks(hand_landmarks):

    x_values = [
        landmark.x
        for landmark in hand_landmarks.landmark
    ]

    y_values = [
        landmark.y
        for landmark in hand_landmarks.landmark
    ]

    min_x = min(x_values)
    min_y = min(y_values)

    data = []

    for landmark in hand_landmarks.landmark:

        data.append(
            landmark.x - min_x
        )

        data.append(
            landmark.y - min_y
        )

    return np.asarray(data, dtype=np.float32)


# ============================================================
# COLOR CONVERSION
# ============================================================

def hex_to_bgr(hex_color):

    hex_color = hex_color.lstrip("#")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return b, g, r


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class GestureProcessor(VideoProcessorBase):

    def __init__(self):

        self.model = load_model()

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.current_gesture = "None"
        self.current_icon = "🖐️"
        self.current_color = "#8b5cf6"
        self.confidence = 0.0

        self.frame_count = 0

        self.lock = threading.Lock()


    def recv(self, frame: av.VideoFrame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        # Mirror webcam
        image = cv2.flip(image, 1)

        height, width, _ = image.shape

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(rgb)

        self.frame_count += 1

        gesture = "None"
        icon = "🖐️"
        color = "#8b5cf6"
        confidence = 0.0


        # ----------------------------------------------------
        # HAND DETECTED
        # ----------------------------------------------------

        if (
            results.multi_hand_landmarks
            and self.model is not None
        ):

            for hand_landmarks in results.multi_hand_landmarks:

                # Draw MediaPipe skeleton
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )


                # Normalize landmarks
                features = normalize_landmarks(
                    hand_landmarks
                )


                # Prediction
                try:

                    prediction = self.model.predict(
                        [features]
                    )[0]

                    class_id = int(prediction)

                    # Confidence
                    if hasattr(
                        self.model,
                        "predict_proba"
                    ):

                        probabilities = (
                            self.model.predict_proba(
                                [features]
                            )[0]
                        )

                        confidence = (
                            float(np.max(probabilities))
                            * 100
                        )

                    else:

                        confidence = 100.0


                    gesture, icon, color = CONTROL_MAP.get(
                        class_id,
                        (
                            "Unknown",
                            "❓",
                            "#94a3b8",
                        )
                    )


                except Exception:

                    gesture = "Prediction Error"
                    icon = "⚠️"
                    color = "#ef4444"
                    confidence = 0.0


                # ------------------------------------------------
                # BOUNDING BOX
                # ------------------------------------------------

                xs = [
                    landmark.x
                    for landmark in hand_landmarks.landmark
                ]

                ys = [
                    landmark.y
                    for landmark in hand_landmarks.landmark
                ]

                x1 = max(
                    int(min(xs) * width) - 15,
                    0
                )

                y1 = max(
                    int(min(ys) * height) - 15,
                    0
                )

                x2 = min(
                    int(max(xs) * width) + 15,
                    width - 1
                )

                y2 = min(
                    int(max(ys) * height) + 15,
                    height - 1
                )


                bgr = hex_to_bgr(color)


                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    bgr,
                    2,
                )


                # Label background
                label = (
                    f"{gesture} "
                    f"{confidence:.0f}%"
                )

                (
                    text_width,
                    text_height
                ), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    2,
                )


                label_y = max(
                    y1 - text_height - 12,
                    0
                )


                cv2.rectangle(
                    image,
                    (
                        x1,
                        label_y
                    ),
                    (
                        x1 + text_width + 12,
                        y1
                    ),
                    bgr,
                    -1,
                )


                cv2.putText(
                    image,
                    label,
                    (
                        x1 + 6,
                        y1 - 7
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )


                break


        # ----------------------------------------------------
        # NO HAND
        # ----------------------------------------------------

        else:

            cv2.putText(
                image,
                "No Hand Detected",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (150, 150, 150),
                2,
                cv2.LINE_AA,
            )


        # ----------------------------------------------------
        # UPDATE SHARED STATE
        # ----------------------------------------------------

        with self.lock:

            self.current_gesture = gesture
            self.current_icon = icon
            self.current_color = color
            self.confidence = confidence


        # ----------------------------------------------------
        # FRAME INFORMATION
        # ----------------------------------------------------

        cv2.putText(
            image,
            f"FRAME  {self.frame_count}",
            (20, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )


        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


if "last_gesture" not in st.session_state:
    st.session_state.last_gesture = "None"


if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🖐️ Gesture AI"
    )

    st.caption(
        "Computer Vision Control System"
    )

    st.divider()


    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    st.markdown(
        "### 🤖 Model"
    )

    model = load_model()


    if model is not None:

        st.success(
            "Model loaded"
        )

    else:

        st.error(
            "model.p not found"
        )

        uploaded_model = st.file_uploader(
            "Upload model.p",
            type=["p"],
        )

        if uploaded_model is not None:

            with open(
                MODEL_PATH,
                "wb"
            ) as file:

                file.write(
                    uploaded_model.getbuffer()
                )

            st.success(
                "Model uploaded. Refreshing..."
            )

            st.cache_resource.clear()

            st.rerun()


    st.divider()


    # --------------------------------------------------------
    # GESTURE MAP
    # --------------------------------------------------------

    st.markdown(
        "### 🎯 Gesture Map"
    )


    for class_id, (
        name,
        icon,
        color
    ) in CONTROL_MAP.items():

        st.markdown(
            f"""
            <div class="gesture-item">
                <span class="gesture-icon">
                    {icon}
                </span>

                <span class="gesture-name">
                    <b style="color:{color}">
                        {class_id}
                    </b>
                    &nbsp; {name}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


    st.divider()


    # --------------------------------------------------------
    # CLEAR HISTORY
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear History",
        use_container_width=True,
    ):

        st.session_state.history = []
        st.session_state.last_gesture = "None"

        st.rerun()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🖐️ Gesture AI
        </div>

        <div class="hero-subtitle">
            Real-time hand gesture recognition powered by
            MediaPipe + Machine Learning
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP STATUS
# ============================================================

if model is not None:

    st.markdown(
        """
        <div class="status">
            <span class="status-dot"></span>
            AI MODEL READY
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.warning(
        "Upload model.p from the sidebar before starting the camera."
    )


st.write("")


# ============================================================
# MAIN COLUMNS
# ============================================================

camera_column, dashboard_column = st.columns(
    [2.2, 1],
    gap="large",
)


# ============================================================
# CAMERA
# ============================================================

with camera_column:

    st.markdown(
        "### 📹 Live Camera"
    )

    st.caption(
        "Allow camera access in your browser, then start the stream."
    )


    if model is None:

        st.info(
            "Your trained model is required to start recognition."
        )

        ctx = None

    else:

        ctx = webrtc_streamer(
            key="gesture-ai",
            video_processor_factory=GestureProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
            async_processing=True,
        )


# ============================================================
# DASHBOARD PLACEHOLDERS
# ============================================================

with dashboard_column:

    st.markdown(
        "### 🎯 Live Result"
    )

    result_placeholder = st.empty()

    st.write("")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:

        confidence_placeholder = st.empty()

    with metric_col2:

        frame_placeholder = st.empty()

    st.write("")

    st.markdown(
        "### 🕓 Detection History"
    )

    history_placeholder = st.empty()


# ============================================================
# LIVE DASHBOARD
#
# IMPORTANT:
# No while True here.
#
# Streamlit fragment periodically reruns this section.
# ============================================================

if ctx is not None:

    @st.fragment(run_every=0.2)
    def live_dashboard():

        processor = ctx.video_processor


        # ----------------------------------------------------
        # CAMERA NOT STARTED
        # ----------------------------------------------------

        if processor is None:

            result_placeholder.markdown(
                """
                <div class="result-card">

                    <div class="result-icon">
                        🖐️
                    </div>

                    <div class="result-name">
                        Ready
                    </div>

                    <div class="result-confidence">
                        Start the camera to begin recognition
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            confidence_placeholder.markdown(
                """
                <div class="metric-card">

                    <div class="metric-number">
                        0%
                    </div>

                    <div class="metric-label">
                        CONFIDENCE
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            frame_placeholder.markdown(
                """
                <div class="metric-card">

                    <div class="metric-number">
                        0
                    </div>

                    <div class="metric-label">
                        FRAMES
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            history_placeholder.markdown(
                """
                <div class="glass-card">

                    <div style="
                        text-align:center;
                        color:#667085;
                        padding:20px;
                    ">
                        No detections yet
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            return


        # ----------------------------------------------------
        # READ PROCESSOR STATE
        # ----------------------------------------------------

        with processor.lock:

            gesture = processor.current_gesture
            icon = processor.current_icon
            color = processor.current_color
            confidence = processor.confidence
            frames = processor.frame_count


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        result_placeholder.markdown(
            f"""
            <div class="result-card"
                 style="border-color:{color}55;">

                <div class="result-icon">
                    {icon}
                </div>

                <div
                    class="result-name"
                    style="color:{color};"
                >
                    {gesture}
                </div>

                <div class="result-confidence">
                    Detection confidence:
                    <b style="color:#e8ebf0">
                        {confidence:.1f}%
                    </b>
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence_placeholder.markdown(
            f"""
            <div class="metric-card">

                <div
                    class="metric-number"
                    style="color:{color};"
                >
                    {confidence:.0f}%
                </div>

                <div class="metric-label">
                    CONFIDENCE
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        # ----------------------------------------------------
        # FRAMES
        # ----------------------------------------------------

        frame_placeholder.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-number">
                    {frames:,}
                </div>

                <div class="metric-label">
                    FRAMES
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        if (
            gesture != "None"
            and gesture != "Prediction Error"
        ):

            # Add only when gesture changes
            if (
                gesture
                != st.session_state.last_gesture
            ):

                st.session_state.history.append(
                    {
                        "name": gesture,
                        "icon": icon,
                        "confidence": confidence,
                        "time": time.strftime(
                            "%H:%M:%S"
                        ),
                    }
                )

                st.session_state.last_gesture = gesture


                # Keep last 10
                if len(
                    st.session_state.history
                ) > 10:

                    st.session_state.history = (
                        st.session_state.history[-10:]
                    )


        # ----------------------------------------------------
        # HISTORY HTML
        # ----------------------------------------------------

        if not st.session_state.history:

            history_html = """
            <div class="glass-card">

                <div style="
                    text-align:center;
                    color:#667085;
                    padding:20px;
                ">
                    No detections yet
                </div>

            </div>
            """

        else:

            history_html = """
            <div class="glass-card">
            """


            for entry in reversed(
                st.session_state.history
            ):

                entry_color = "#8b5cf6"


                for (
                    _,
                    (
                        mapped_name,
                        _,
                        mapped_color,
                    ),
                ) in CONTROL_MAP.items():

                    if mapped_name == entry["name"]:

                        entry_color = mapped_color

                        break


                history_html += f"""
                <div class="history-row">

                    <span>
                        {entry["icon"]}
                        <span style="
                            color:{entry_color};
                            font-weight:600;
                        ">
                            {entry["name"]}
                        </span>
                    </span>

                    <span class="history-time">
                        {entry["confidence"]:.0f}%
                        ·
                        {entry["time"]}
                    </span>

                </div>
                """


            history_html += "</div>"


        history_placeholder.markdown(
            history_html,
            unsafe_allow_html=True,
        )


    live_dashboard()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br>

    <div style="
        text-align:center;
        color:#4b5563;
        font-size:11px;
        padding:20px;
    ">

        Gesture AI · MediaPipe · Machine Learning · Streamlit

    </div>
    """,
    unsafe_allow_html=True,
)
```

### Your `requirements.txt`

Use this exact file:

```text
streamlit
opencv-python-headless
mediapipe
numpy
scikit-learn
streamlit-webrtc
av
```

### Repository structure

```text
hand-gesture-smart-control-using-mediapipe-machine-learning/
│
├── app.py
├── model.p
├── requirements.txt
└── README.md
```

### One important change from your original project

Your original app had:

```python
while True:
    ...
    time.sleep(0.1)
```

I removed that completely.

Instead, the dashboard uses Streamlit's periodic fragment:

```python
@st.fragment(run_every=0.2)
def live_dashboard():
```

