import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Happy or Sad Classifier", layout="wide", page_icon="😊")

st.sidebar.image("pngwing.com (12).png", width=300)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
selected_page = st.sidebar.radio("Navigation", ["Home", "Classify"])

# ===================== LOAD MODEL =====================
MODEL_PATH = os.path.join("models", "imageclassifier2.h5")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at '{MODEL_PATH}'. Please ensure the model is in the /models directory.")
        st.stop()
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

model = load_model()

# ===================== CLASSIFY IMAGE =====================
def classify_image(image):
    """
    image: numpy array of shape (H, W, 3)
    returns: (label, confidence_pct)
    """
    resized = tf.image.resize(image, (256, 256)).numpy()
    normalized = np.expand_dims(resized / 255.0, 0)
    yhat = model.predict(normalized, verbose=0)[0][0]

    if yhat > 0.5:
        label = "Sad"
        confidence = yhat * 100
    else:
        label = "Happy"
        confidence = (1 - yhat) * 100

    return label, round(float(confidence), 1)

# ===================== HOME PAGE =====================
def HomePage():
    st.markdown("""
        <h1 style='color:#2B2A4C; text-align:center; font-family: Georgia, serif;'>
            😊 Happy or Sad Image Classifier
        </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
        <p style='text-align:center; color:#555; font-size:17px; max-width:700px; margin:auto;'>
            A deep learning model that analyses facial expressions in real-time and classifies
            them as <strong>Happy</strong> or <strong>Sad</strong>. Upload a photo or use your 
            webcam to try it out.
        </p>
        <br>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div style='background:#f0f4ff; border-radius:12px; padding:20px; text-align:center;'>
                <h3>🧠 Deep Learning</h3>
                <p>Powered by a custom Convolutional Neural Network (CNN) trained on facial expression data.</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div style='background:#fff0f4; border-radius:12px; padding:20px; text-align:center;'>
                <h3>📷 Real-Time</h3>
                <p>Use your webcam for live classification or upload an image to test the model instantly.</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div style='background:#f0fff4; border-radius:12px; padding:20px; text-align:center;'>
                <h3>📊 Confidence Score</h3>
                <p>Get the model's confidence percentage alongside every prediction.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <h3 style='color:#2B2A4C;'>How It Works</h3>
        <ol style='font-size:16px; line-height:2;'>
            <li>The model was trained on images of happy and sad faces.</li>
            <li>Input images are resized to 256x256 pixels and normalised.</li>
            <li>The CNN outputs a score between 0 (Happy) and 1 (Sad).</li>
            <li>Confidence is derived from how far the score is from 0.5.</li>
        </ol>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:left; font-size:12px; color:#aaa;'>Created with ❤️ by Datapsalm</p>", unsafe_allow_html=True)

# ===================== VIDEO PROCESSOR =====================
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        import av
        image = frame.to_ndarray(format="bgr24")
        resized = cv2.resize(image, (256, 256))
        prediction, confidence = classify_image(resized)

        color = (0, 200, 100) if prediction == "Happy" else (0, 80, 220)
        label_text = f"{prediction}  ({confidence}%)"

        cv2.putText(image, label_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# ===================== CLASSIFY PAGE =====================
def ClassifyPage():
    st.markdown("<h2 style='color:#2B2A4C;'>Happy or Sad Classifier</h2>", unsafe_allow_html=True)

    option = st.radio("Choose an option:", ("Upload Image", "Live Webcam"))

    # ----------- UPLOAD IMAGE -----------
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                st.error("Could not read the image. Please upload a valid JPG or PNG file.")
                return

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

            with st.spinner("Analysing..."):
                prediction, confidence = classify_image(image_rgb)

            emoji = "😊" if prediction == "Happy" else "😢"
            color = "#1a7a4a" if prediction == "Happy" else "#c0392b"

            st.markdown(f"""
                <div style='background:#f8f8f8; border-left: 5px solid {color};
                     border-radius:8px; padding:20px; margin-top:16px;'>
                    <h2 style='color:{color}; margin:0;'>{emoji} {prediction}</h2>
                    <p style='font-size:16px; color:#555; margin-top:8px;'>
                        Confidence: <strong>{confidence}%</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)

            st.progress(int(confidence))

    # ----------- LIVE WEBCAM -----------
    elif option == "Live Webcam":
        st.info("Allow camera access when prompted. Predictions will appear on the live feed.")
        webrtc_streamer(
            key="happy-sad-classifier",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

# ===================== PAGE ROUTING =====================
if selected_page == "Home":
    HomePage()
elif selected_page == "Classify":
    ClassifyPage()
