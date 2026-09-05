import streamlit as st
import os
import time

# ── Handle OpenCV import gracefully ──────────────────────────────────────────
try:
    import cv2
except Exception as e:
    st.error(f"Failed to import OpenCV: {e}")
    st.info("This usually means system libraries are missing. Make sure packages.txt is in your repo root.")
    st.stop()

import mediapipe as mp
import numpy as np
import pickle
import av

# ── Optional: pyautogui only works when running LOCALLY ─────────────────────
try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except Exception:
    PYAUTOGUI_AVAILABLE = False

# ── Load model ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.p")
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model_data = pickle.load(open(MODEL_PATH, "rb"))

# Handle both save formats
if isinstance(model_data, dict):
    model = model_data["model"]
else:
    model = model_data

# ── Debug: print model classes on startup ───────────────────────────────────
try:
    print(f"Model classes: {model.classes_}")
except Exception:
    print("Model has no classes_ attribute")

# ── Gesture label map ────────────────────────────────────────────────────────
LABEL_MAP = {
    "0": "Volume Up",
    "1": "Volume Down",
    "2": "Play / Pause",
    "3": "Previous",
    "4": "Next",
    "5": "Mute",
    "6": "Scroll Up",
    "7": "Scroll Down",
    "8": "Brightness Up",
    "9": "Brightness Down",
}

# ── Gesture → action mapping (only works locally) ────────────────────────────
ACTIONS = {
    "0": lambda: pyautogui.press("volumeup") if PYAUTOGUI_AVAILABLE else None,
    "1": lambda: pyautogui.press("volumedown") if PYAUTOGUI_AVAILABLE else None,
    "2": lambda: pyautogui.press("playpause") if PYAUTOGUI_AVAILABLE else None,
    "3": lambda: pyautogui.hotkey("alt", "left") if PYAUTOGUI_AVAILABLE else None,
    "4": lambda: pyautogui.hotkey("alt", "right") if PYAUTOGUI_AVAILABLE else None,
    "5": lambda: pyautogui.press("volumemute") if PYAUTOGUI_AVAILABLE else None,
    "6": lambda: pyautogui.scroll(3) if PYAUTOGUI_AVAILABLE else None,
    "7": lambda: pyautogui.scroll(-3) if PYAUTOGUI_AVAILABLE else None,
    "8": lambda: pyautogui.hotkey("win", "a") if PYAUTOGUI_AVAILABLE else None,
    "9": lambda: pyautogui.hotkey("win", "a") if PYAUTOGUI_AVAILABLE else None,
}

# ── Video processor ───────────────────────────────────────────────────────────
class GestureProcessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.label = ""
        self.confidence = 0.0
        self.last_action = 0.0
        self.cooldown = 1.5

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw skeleton
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )

                # Bounding box
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                cv2.rectangle(
                    img,
                    (int(x_min * w) - 20, int(y_min * h) - 20),
                    (int(x_max * w) + 20, int(y_max * h) + 20),
                    (0, 255, 0), 2,
                )

                # Extract 42 features (normalised)
                features = []
                for lm in hand_landmarks.landmark:
                    features.append(lm.x - x_min)
                    features.append(lm.y - y_min)

                # Predict
                features = np.array(features).reshape(1, -1)
                prediction = model.predict(features)[0]
                self.confidence = model.predict_proba(features).max()
                self.label = str(prediction)

                # Execute action (with cooldown + confidence threshold)
                now = time.time()
                if (
                    self.label in ACTIONS
                    and self.confidence >= 0.80
                    and (now - self.last_action) > self.cooldown
                    and PYAUTOGUI_AVAILABLE
                ):
                    try:
                        ACTIONS[self.label]()
                        self.last_action = now
                    except Exception:
                        pass

                # Overlay info
                action_name = LABEL_MAP.get(self.label, self.label)
                cv2.rectangle(img, (0, 0), (420, 100), (0, 0, 0), -1)
                cv2.putText(
                    img,
                    f"Gesture: {action_name}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    f"Confidence: {self.confidence:.0%}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
        else:
            cv2.putText(
                img,
                "No hand detected",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Hand Gesture Control", page_icon="🖐️")
st.title("🖐️ Hand Gesture Smart Control")

if not PYAUTOGUI_AVAILABLE:
    st.info("💡 Running on Streamlit Cloud — gestures are detected and displayed, but system actions (volume, mouse, etc.) only work when running **locally**.")

st.write("Show a hand gesture to the camera to see real-time recognition.")

# ── Gesture reference table ───────────────────────────────────────────────────
st.subheader("📋 Gesture Reference")
st.table({
    "Gesture #": list(LABEL_MAP.keys()),
    "Action": list(LABEL_MAP.values()),
})

st.divider()

# ── WebRTC stream ─────────────────────────────────────────────────────────────
from streamlit_webrtc import webrtc_streamer

webrtc_streamer(
    key="gesture",
    video_processor_factory=GestureProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.caption("Built with Streamlit + MediaPipe + scikit-learn")
