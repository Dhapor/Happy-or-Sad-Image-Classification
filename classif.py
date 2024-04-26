
# import streamlit as st
# import tensorflow as tf


# # @st.cache_data(allow_output_mutation=True)
# def load_model():
#   model=tf.keras.models.load_model('models\happysadmodel.h5')
#   return model
# with st.spinner('Model is being loaded..'):
#   model=load_model()

# st.write("""
#          # Flower Classification
#          """
#          )

# file = st.file_uploader("Please upload an image", type=["jpg", "png", "bmp", "jpeg"])
# import cv2
# from PIL import Image, ImageOps
# import numpy as np
# st.set_option('deprecation.showfileUploaderEncoding', False)
# def import_and_predict(image_data, model):
    
#         size = (180,180)    
#         image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
#         image = np.asarray(image)
#         img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

#         img_reshape = img[np.newaxis,...]
#         prediction = model.predict(img_reshape)
#         return prediction

# if file is None:
#     st.text("Please upload an image file")
# else:
#     image = Image.open(file)
#     st.image(image, use_column_width=True)
#     predictions = import_and_predict(image, model)
#     score = tf.nn.softmax(predictions[0])
#     st.write(prediction)
#     st.write(score)
#     print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )






import streamlit as st
# import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt




# Load the pre-trained model
model = tf.keras.models.load_model('models\imageclassifier2.h5')


st.sidebar.image('pngwing.com (12).png', width = 300,)
st.sidebar.markdown('<br>', unsafe_allow_html=True)
selected_page = st.sidebar.radio('Navigation', ['Home', 'Modeling'])

def HomePage():
    # Streamlit app header
    st.markdown("<h1 style = 'color: #2B2A4C; text-align: center; font-family:montserrat'>Image Classification Model</h1>",unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h6 style = 'margin: -15px; color: #2B2A4C; text-align: center ; font-family:montserrat'>This is a Diabetes Prediction Model that was built Using Machine Learning to Enhance Early Detection and Improve Patient Outcomes.</h6>",unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    # st.image('newk.svg',  width = 700)
    st.markdown('<br>', unsafe_allow_html= True)


 # Background story
    st.markdown("<h3 style = 'margin: -15px; color: #2B2A4C; text-align: center; font-family:montserrat'>Background to the story</h3>",unsafe_allow_html=True)
    st.markdown("This project is personal to me because my grandpa had diabetes for a long time. I want to create a computer program that can tell if someone might get diabetes in the future. By using information about a person's health, the program will try to help them know if they need to be careful. I'm doing this so that others don't have to go through what my grandpa did. Let's work together to use technology to help people stay healthy and avoid diabetes", unsafe_allow_html = True)

    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)

    st.markdown("<h3 style='color: #2B2A4C;text-align: center; font-family:montserrat'>The Model Features</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Gender</h3>", unsafe_allow_html=True)
    st.markdown("<p>Gender refers to the biological sex of the individual, which can have an impact on their susceptibility to diabetes. There are three</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Age</h3>", unsafe_allow_html=True)
    st.markdown("<p>Age is an important factor as diabetes is more commonly diagnosed in older adults.Age ranges from 0-80 in our dataset.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Hypertension</h3>", unsafe_allow_html=True)
    st.markdown("<p>Hypertension is a medical condition in which the blood pressure in the arteries is persistently elevated.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Heart Diseases</h3>", unsafe_allow_html=True)
    st.markdown("<p>Heart disease is another medical condition that is associated with an increased risk of developing diabetes.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Smoking history</h3>", unsafe_allow_html=True)
    st.markdown("<p>Smoking history is also considered a risk factor for diabetes and can exacerbate the complications associated</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Body Mass Index</h3>", unsafe_allow_html=True)
    st.markdown("<p>BMI (Body Mass Index) is a measure of body fat based on weight and height. Higher BMI values are linked to a higher risk</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Hemoglobin A1c</h3>", unsafe_allow_html=True)
    st.markdown("<p>HbA1c (Hemoglobin A1c) level is a measure of a person's average blood sugar level over the past 2-3 months.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Blood glucose level</h3>", unsafe_allow_html=True)
    st.markdown("<p>Blood glucose level refers to the amount of glucose in the bloodstream at a given time. </p>", unsafe_allow_html=True)


    # Streamlit app footer
    st.markdown("<p style='text-align: LEFT; font-size: 12px;'>Created with ❤️ by Datapsalm</p>", unsafe_allow_html=True)

# Function to define the modeling page content
def modeling_page():
    st.markdown("<h1 style='text-align: LEFT; color: #2B2A4C;'></h1>", unsafe_allow_html=True)
    # st.sidebar.markdown('<br><br><br>', unsafe_allow_html= True)
    # st.write(df.head())
    # st.sidebar.image('pngwing.com (13).png', width = 300,  caption = 'customer and deliver agent info')


if selected_page == "Home":
    HomePage()
elif selected_page == "Modeling":
    st.sidebar.markdown('<br>', unsafe_allow_html= True)
    modeling_page()


if selected_page == "Modeling":
    # Define a class for video transformation
    from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        resized_image = cv2.resize(image, (256, 256))
        resized_image = tf.image.resize(resized_image, (256, 256))
        prediction = classify_image(resized_image)
        cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return image

# Function to perform image classification
def classify_image(image):
    # Perform prediction
    yhat = model.predict(np.expand_dims(image / 255, 0))

    # Determine the predicted label
    if yhat > 0.5: 
        return "Sad"
    else:
        return "Happy"

# Streamlit app
st.title("Happy or Sad Image Classification")

# Option to choose between live capturing and file upload
option = st.radio("Choose an option:", ("Live Capture", "Upload Image"))

if option == "Live Capture":
    # Display live camera feed and perform image classification
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

elif option == "Upload Image":
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Display the original image
        st.image(image, caption="Original Image", use_column_width=True)

        # Perform image classification
        resized_image = tf.image.resize(image, (256, 256))
        prediction = classify_image(resized_image)

        # Display the prediction
        st.write(prediction)


st.markdown("<h8 style = 'color: #2B2A4C; text-align: LEFT; font-family:montserrat'>DIABETES PREDICTION MODEL BUILT BY DATAPSALM</h8>",unsafe_allow_html=True)
