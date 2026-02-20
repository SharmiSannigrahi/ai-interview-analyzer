import streamlit as st
from sentiment import analyze_sentiment
from transformers import pipeline

st.set_page_config(page_title="AI Interview Analyzer")

st.title("🎯 AI Interview Analyzer")
st.write("Analyze your confidence using AI")

# -----------------------
# TEXT SENTIMENT
# -----------------------

user_text = st.text_area("🎤 Enter your interview answer:")

if st.button("Analyze Text"):
    if user_text:
        sentiment_result = analyze_sentiment(user_text)
        st.success(f"Sentiment: {sentiment_result['label']}")
        st.write(f"Confidence Score: {round(sentiment_result['score']*100)}%")
    else:
        st.warning("Please enter text.")

# -----------------------
# FACIAL EMOTION (LIGHT VERSION)
# -----------------------

st.subheader("😊 Emotion from Text Tone")

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

if st.button("Analyze Emotion"):
    if user_text:
        emotion = emotion_classifier(user_text)
        st.success(f"Detected Emotion: {emotion[0]['label']}")