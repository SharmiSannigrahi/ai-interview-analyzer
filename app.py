import streamlit as st
from sentiment import analyze_sentiment
from deepface import DeepFace
import cv2
import numpy as np

st.set_page_config(page_title="AI Interview Analyzer", layout="centered")

st.title("🎯 AI Interview Analyzer")
st.write("Analyze your confidence using AI")

# ----------------------------
# TEXT INPUT (Speech alternative for web)
# ----------------------------

user_text = st.text_area("🎤 Enter your interview answer:")

if st.button("Analyze Text Sentiment"):
    if user_text:
        sentiment_result = analyze_sentiment(user_text)
        st.success(f"Sentiment: {sentiment_result['label']}")
        st.write(f"Confidence Score: {round(sentiment_result['score']*100)}%")
    else:
        st.warning("Please enter some text.")

# ----------------------------
# IMAGE EMOTION DETECTION
# ----------------------------

st.subheader("😊 Upload your face image")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, channels="BGR")

    result = DeepFace.analyze(img, actions=['emotion'])
    emotion = result[0]['dominant_emotion']

    st.success(f"Detected Emotion: {emotion}")