# ✋ Hand Gesture Smart Control using MediaPipe & Machine Learning

🚀 A real-time AI hand gesture recognition system that allows users to control their computer using hand gestures detected through a webcam.

This project combines:
- 🤖 Artificial Intelligence
- 👁️ Computer Vision
- 🧠 Machine Learning
- 💻 Human-Computer Interaction (HCI)
- ⚡ Real-time Processing

Using MediaPipe hand tracking and a Random Forest classifier, the system can recognize multiple gestures and execute computer actions such as:
- 🖱️ Mouse movement
- 👆 Left click
- 👉 Right click
- 🔊 Volume control
- 📜 Scrolling
- 📸 Screenshot capture

---

# ✨ Features

- ✋ Real-time hand tracking
- 🤖 AI-based gesture recognition
- 🖱️ Mouse movement control
- 👆 Left click and 👉 right click
- 📜 Scroll functionality
- 🔊 Volume up/down control
- 📸 Screenshot capture
- ⚡ Smooth mouse movement
- 📂 Custom dataset collection
- 🧠 Machine learning model training

---

# 🛠️ Technologies Used

- 🐍 Python
- 👁️ OpenCV
- ✋ MediaPipe
- 🧠 Scikit-learn
- 🔢 NumPy
- 💻 PyAutoGUI

---

# 📁 Project Structure

```bash
├── collect_images.py
├── create_dataset.py
├── train_classifier.py
├── inference_classifier.py
├── model.p
├── data.pickle
├── dataset/
└── README.md
```

---

# ⚙️ Installation

## 📥 Clone Repository

```bash
git clone https://github.com/yourusername/hand-gesture-smart-control.git
cd hand-gesture-smart-control
```

## 📦 Install Dependencies

```bash
pip install opencv-python mediapipe scikit-learn numpy pyautogui
```

---

# 🚀 Usage

## 1️⃣ Collect Dataset

Run:

```bash
python collect_images.py
```

### 🎮 Controls
- `s` → start capture
- `q` → stop program

---

## 2️⃣ Create Dataset

Run:

```bash
python create_dataset.py
```

📌 This step extracts hand landmarks using MediaPipe and creates the dataset file.

---

## 3️⃣ Train Model

Run:

```bash
python train_classifier.py
```

🧠 This trains the Random Forest classifier and saves the trained model.

---

## 4️⃣ Start Gesture Control

Run:

```bash
python inference_classifier.py
```

### 🎮 Controls
- `q` → quit application

---

# ✋ Supported Gestures

| Gesture Class | Action |
|---|---|
| 0 | 🖱️ Move Mouse |
| 1 | ✋ Stop Mouse |
| 2 | 👆 Left Click |
| 3 | 👉 Right Click |
| 4 | 📜 Scroll |
| 5 | 🔊 Volume Up |
| 6 | 🔉 Volume Down |
| 7 | 📸 Screenshot |

---

# 🧠 Machine Learning Pipeline

1. 📂 Collect gesture images
2. ✋ Extract hand landmarks using MediaPipe
3. 📐 Normalize landmark coordinates
4. 🧠 Train Random Forest classifier
5. ⚡ Predict gestures in real-time
6. 💻 Execute computer actions

---

# 🔮 Future Improvements

- 🧠 Deep Learning model (CNN/LSTM)
- ✨ Dynamic gesture recognition
- ✋✋ Multi-hand support
- ⚙️ Gesture customization
- 🖥️ GUI dashboard
- 🎙️ Voice assistant integration
- 🏠 Smart home control

---

# 🌍 Applications

- 💻 Touchless computer interaction
- ♿ Accessibility systems
- 🏠 Smart environments
- 🖱️ Virtual mouse systems
- 🤖 AI-based automation
- 🧪 Human-computer interaction research

---

# 👨‍💻 Author

Developed by **Khalil Nouar**

---

# 📜 License

This project is licensed under the **MIT License**.
