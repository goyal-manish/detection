import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Smile Sad Surprise Detector", page_icon="😄")

st.title("😄 😢 😲 Face Emotion Detector")
st.write("Detect Smile, Sad & Surprise from Image")

# Load Haar Cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

uploaded_file = st.file_uploader("Upload Image 📸", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.error("❌ No face detected")
    else:
        for i, (x, y, w, h) in enumerate(faces, start=1):
            face_gray = gray[y:y+h, x:x+w]
            face_color = img[y:y+h, x:x+w]

            # Detect smile
            smiles = smile_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.7,
                minNeighbors=20
            )

            # Mouth region (lower half of face)
            mouth_region = face_color[int(h*0.6):h, :]
            hsv = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2HSV)
            mouth_open = np.mean(hsv[:, :, 2])

            # Emotion decision
            if len(smiles) > 0:
                emotion = "😄 Smile"
                color = (0, 255, 0)
            elif mouth_open > 150:
                emotion = "😲 Surprise"
                color = (255, 255, 0)
            else:
                emotion = "😢 Sad"
                color = (0, 0, 255)

            # Draw rectangle & label
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
            cv2.putText(
                img,
                f"Face {i}: {emotion}",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        st.image(img, caption="Detected Emotions", use_column_width=True)
        st.success("🎉 Emotion detection completed!")
