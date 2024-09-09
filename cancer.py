import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.title("Lung Cancer Detection")

uploaded_file = st.file_uploader("Choose a file for Lung Cancer Detection", type=["jpg", "png", "jpeg"])

def load_model():
    model = tf.keras.models.load_model('lung_cancer_model.keras')
    return model

def predict(image, model):
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.LANCZOS)  # Use Image.LANCZOS instead of Image.ANTIALIAS
    image = image.convert('RGB')  # Ensure the image has 3 color channels (RGB)
    img = np.asarray(image)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    model = load_model()
    prediction = predict(image, model)
    
    class_names = ['Begin Case', 'Normal Case', 'Malignant Case']
    predicted_class = class_names[np.argmax(prediction)]
    st.write(f"Prediction: {predicted_class}")