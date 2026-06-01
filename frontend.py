import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
from ultralytics import YOLO
from gtts import gTTS
import playsound
import os
import time

st.set_page_config(page_title="Sign Language Detection", layout="wide")
st.title("Automated Sign Language Detection with Voice Output")

@st.cache_resource
def load_models():
    yolo = YOLO("yolov8n.pt")
    cnn_model = keras.models.load_model("sign_lang_model.h5")
    return yolo, cnn_model

yolo, cnn_model = load_models()
labels = [chr(i) for i in range(65, 91)]

if "sentence" not in st.session_state:
    st.session_state.sentence = ""
if "last_pred" not in st.session_state:
    st.session_state.last_pred = ""
if "last_time" not in st.session_state:
    st.session_state.last_time = time.time()

col1, col2 = st.columns([3, 1])

start_cam = col1.button("🎥 Start Webcam")
stop_cam = col1.button("🛑 Stop Webcam")
clear_text = col2.button("🧹 Clear Text")
speak_now = col2.button("🔊 Speak Detected Word")

FRAME_WINDOW = col1.image([])

def speak_text(text):
    tts = gTTS(text=text, lang="en")
    filename = "temp_audio.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)

if start_cam:
    st.session_state.running = True

if stop_cam:
    st.session_state.running = False

if "running" not in st.session_state:
    st.session_state.running = False

cap = cv2.VideoCapture(0)

while st.session_state.running:
    ret, frame = cap.read()
    if not ret:
        st.warning("Camera not detected.")
        break

    results = yolo(frame)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            hand = frame[y1:y2, x1:x2]
            if hand.size == 0:
                continue

            hand = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
            hand = cv2.resize(hand, (28, 28))
            hand = hand / 255.0
            hand = hand.reshape(1, 28, 28, 1)

            predictions = cnn_model.predict(hand, verbose=0)
            pred_idx = np.argmax(predictions)
            label = labels[pred_idx]
            confidence = np.max(predictions) * 100

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.1f}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if time.time() - st.session_state.last_time > 1:
                if label != st.session_state.last_pred:
                    st.session_state.sentence += label
                    st.session_state.last_pred = label

                st.session_state.last_time = time.time()

    FRAME_WINDOW.image(frame, channels="BGR")
    st.write("**Detected Word:**", st.session_state.sentence)

    if clear_text:
        st.session_state.sentence = ""

    if speak_now and st.session_state.sentence:
        speak_text(st.session_state.sentence)

    st.experimental_rerun()

cap.release()
