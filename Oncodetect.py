import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.title("OncoDetect")

uploaded_file=st.file_uploader("Choose a file for Lung Cancer Detection", type=["jpg", "png", "jpeg"])

uploaded_file1=st.file_uploader("Choose a file for Brain tumor Detection", type=["jpg", "png", "jpeg"])

def load_lung_model():
    lung_model = tf.keras.models.load_model('lung_cancer2.h5')
    return lung_model

def load_brain_model():
    brain_model = tf.keras.models.load_model('brain_tumor_model.h5')
    return brain_model

def predict_lung(image,lung_model):
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.LANCZOS)  # Use Image.LANCZOS instead of Image.ANTIALIAS
    image = image.convert('RGB')  # Ensure the image has 3 color channels (RGB)
    img = np.asarray(image)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    lung_prediction = lung_model.predict(img)
    return lung_prediction

def predict_brain(image,brain_model):
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.LANCZOS)  # Use Image.LANCZOS instead of Image.ANTIALIAS
    image = image.convert('RGB')  # Ensure the image has 3 color channels (RGB)
    img = np.asarray(image)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    brain_prediction = brain_model.predict(img)
    return brain_prediction

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    model = load_lung_model()
    prediction = predict_lung(image, model)
    
    class_names = ['Begin Case', 'Normal Case', 'Malignant Case']
    predicted_class = class_names[np.argmax(prediction)]
    st.write(f"Prediction: {predicted_class}")
    
elif uploaded_file1 is not None:
    image = Image.open(uploaded_file1)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    model = load_brain_model()
    prediction = predict_brain(image, model)
    
    optimal_threshold = 0.5  # Replace with the optimal threshold determined from evaluation
    if prediction >= optimal_threshold:
        st.write("Prediction: Positive for Brain Tumor")
    else:
        st.write("Prediction: Negative for Brain Tumor")