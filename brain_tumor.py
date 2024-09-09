import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.title("Brain Tumor Detection")

uploaded_file = st.file_uploader("Choose a file for Brain Tumor Detection", type=["jpg", "png", "jpeg"])

def load_model():
    model = tf.keras.models.load_model('brain_tumor_model.h5')
    return model

def predict(image, model):
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.LANCZOS)  # Use Image.LANCZOS instead of Image.ANTIALIAS
    image = image.convert('RGB')  # Ensure the image has 3 color channels (RGB)
    img = np.asarray(image)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction[0][0]

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    model = load_model()
    prediction = predict(image, model)
    
    optimal_threshold = 0.9597  # Replace with the optimal threshold determined from evaluation
    if prediction >= optimal_threshold:
        st.write("Prediction: Positive for Brain Tumor")
    else:
        st.write("Prediction: Negative for Brain Tumor")