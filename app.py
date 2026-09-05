import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyautogui
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import os

# ── Load model ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.p")
model_data = pickle.load(open(MODEL_PATH, "rb"))

# Handle both save formats
if isinstance(model_data, dict):
    model = model_data["model"]
else:
    model = model_data

# ── Debug: print model classes on startup ───────────────────────────────────
try:
    print(f"Model classes: {model.classes_}")
except:
    print("Model has no classes_ attribute")

# ── Gesture label map (fix inversion here) ───────────────────────────────────
# If gesture "0" triggers wrong action, swap the keys below
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

# ── Gesture → action mapping ─────────────────────────────────────────────────
ACTIONS = {
    "0": lambda: pyautogui.press("volumeup"),
    "1": lambda: pyautogui.press("volumedown"),
    "2": lambda: pyautogui.press("playpause"),
    "3": lambda: pyautogui.hotkey("alt", "left"),
    "4": lambda: pyautogui.hotkey("alt", "right"),
    "5": lambda: pyautogui.press("volumemute"),
    "6": lambda: pyautogui.scroll(3),
    "7": lambda: pyautogui.scroll(-3),
    "8": lambda: pyautogui.hotkey("win", "a"),    # opens action center
    "9": lambda: pyautogui.hotkey("win", "a"),    # same, customize as needed
}

# ── Cooldown to avoid spamming actions ───────────────────────────────────────
import time
COOLDOWN = 1.5   # seconds between actions

# ── Video processor ───────────────────────────────────────────────────────────
class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands     = mp.solutions.hands
        self.hands        = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.mp_draw      = mp.solutions.drawing_utils
        self.label        = ""
        self.last_action  = 0.0   # timestamp of last triggered action

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img     = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # ── Draw skeleton ─────────────────────────────────────────
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )

                # ── Bounding box ──────────────────────────────────────────
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # Draw bounding box
                cv2.rectangle(
                    img,
                    (int(x_min * w) - 20, int(y_min * h) - 20),
                    (int(x_max * w) + 20, int(y_max * h) + 20),
                    (0, 255, 0), 2,
                )

                # ── Extract 42 features (normalised) ──────────────────────
                features = []
                for lm in hand_landmarks.landmark:
                    features.append(lm.x - x_min)
                    features.append(lm.y - y_min)

                # ── Predict ───────────────────────────────────────────────
                features    = np.array(features).reshape(1, -1)
                prediction  = model.predict(features)[0]
                confidence  = model.predict_proba(features).max()
                self.label  = str(prediction)

                print(f"Predicted: {self.label} | Confidence: {confidence:.2f}")

                # ── Execute action (with cooldown + confidence threshold) ──
                now = time.time()
                if (
                    self.label in ACTIONS
                    and confidence >= 0.80
                    and (now - self.last_action) > COOLDOWN
                ):
                    ACTIONS[self.label]()
                    self.last_action = now

                # ── Overlay info on frame ─────────────────────────────────
                action_name = LABEL_MAP.get(self.label, self.label)

                # Background box for text
                cv2.rectangle(img, (0, 0), (400, 90), (0, 0, 0), -1)

                cv2.putText(
                    img,
                    f"Gesture : {action_name}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    f"Confidence: {confidence:.0%}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )

        else:
            # No hand detected
            cv2.putText(
                img,
                "No hand detected",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Hand Gesture Control", page_icon="🖐️")
st.title("🖐️ Hand Gesture Smart Control")
st.write("Show a hand gesture to the camera to trigger system actions.")

# ── Gesture reference table ───────────────────────────────────────────────────
st.subheader("📋 Gesture Reference")
st.table({
    "Gesture #": list(LABEL_MAP.keys()),
    "Action"   : list(LABEL_MAP.values()),
})

st.divider()

# ── Cooldown slider ───────────────────────────────────────────────────────────
COOLDOWN = st.slider(
    "Action cooldown (seconds)",
    min_value=0.5,
    max_value=5.0,
    value=1.5,
    step=0.5,
    help="Prevents the same action from firing too fast",
)

st.divider()

# ── WebRTC stream ─────────────────────────────────────────────────────────────
webrtc_streamer(
    key="gesture",
    video_processor_factory=GestureProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.caption("⚠️ Make sure to run this with Python 3.11 inside a virtual environment.")