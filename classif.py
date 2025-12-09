import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Image Classification App", layout="wide")
st.sidebar.image("pngwing.com (12).png", width=300)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
selected_page = st.sidebar.radio("Navigation", ["Home", "Modeling"])

# ===================== LOAD MODEL =====================
MODEL_PATH = os.path.join("models", "imageclassifier2.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# ===================== CLASSIFY IMAGE =====================
def classify_image(image):
    """
    image: numpy array of shape (256,256,3)
    returns: prediction label "Happy" or "Sad"
    """
    image = np.expand_dims(image / 255.0, 0)  # normalize & add batch dim
    yhat = model.predict(image)
    return "Sad" if yhat[0][0] > 0.5 else "Happy"

# ===================== HOME PAGE =====================
def HomePage():
    st.markdown("<h1 style='color:#2B2A4C;text-align:center;font-family:montserrat'>Image Classification Model</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#2B2A4C'>This is a Diabetes Prediction Model built with Machine Learning to Enhance Early Detection.</p>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#2B2A4C'>Background Story</h3>", unsafe_allow_html=True)
    st.markdown("Personal motivation: my grandpa had diabetes. This program helps predict early risk using health data.", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#2B2A4C'>Model Features</h3>", unsafe_allow_html=True)
    st.markdown("""
    - Gender  
    - Age  
    - Hypertension  
    - Heart Diseases  
    - Smoking History  
    - BMI  
    - Hemoglobin A1c  
    - Blood Glucose Level
    """, unsafe_allow_html=True)
    st.markdown("<p style='text-align:left;font-size:12px'>Created with ❤️ by Datapsalm</p>", unsafe_allow_html=True)

# ===================== VIDEO TRANSFORMER =====================
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        resized_image = cv2.resize(image, (256, 256))
        resized_image = tf.image.resize(resized_image, (256, 256)).numpy()
        prediction = classify_image(resized_image)
        cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return image

# ===================== MODELING PAGE =====================
def ModelingPage():
    st.title("Happy or Sad Image Classification")

    option = st.radio("Choose an option:", ("Live Capture", "Upload Image"))

    # ----------- LIVE WEBCAM -----------
    if option == "Live Capture":
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    # ----------- UPLOAD IMAGE -----------
    elif option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            st.image(image, caption="Original Image", use_column_width=True)
            resized_image = tf.image.resize(image, (256, 256)).numpy()
            prediction = classify_image(resized_image)
            st.markdown(f"<h2 style='color:#2B2A4C'>Prediction: {prediction}</h2>", unsafe_allow_html=True)

# ===================== PAGE SELECTION =====================
if selected_page == "Home":
    HomePage()
elif selected_page == "Modeling":
    ModelingPage()
