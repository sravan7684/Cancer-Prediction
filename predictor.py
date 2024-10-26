import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageOps
import numpy as np
import cv2

st.title("Cancer Predictor")

uploaded_file=st.file_uploader("Choose a file for Lung Cancer Detection", type=["jpg", "png", "jpeg"])

uploaded_file1=st.file_uploader("Choose a file for Brain tumor Detection", type=["jpg", "png", "jpeg"])

def load_lung_model():
    model_path = 'Models/lung_cancer.h5'
    lung_model = tf.keras.models.load_model(model_path)
    return lung_model

def load_brain_model():
    model_path = 'Models/brain_tumor_model.h5'
    brain_model = tf.keras.models.load_model(model_path)
    return brain_model


def predict_lung(model, uploaded_file, target_size=(256, 256), class_labels=['Benign case', 'Malignant case', 'Normal case']):
    
    # Check if the uploaded file is a file-like object or an already processed image
    if hasattr(uploaded_file, 'read'):  # If it's a file-like object
        # Open the image file using PIL
        img = Image.open(uploaded_file).convert('L')  # Convert image to grayscale
    else:
        # If it's already an image, no need to reopen
        img = uploaded_file.convert('L')  # Ensure it’s grayscale

    # Convert the PIL image to a NumPy array
    img_np = np.array(img)

    if img_np is None:
        raise ValueError("Unable to load image")

    # Resize the image to the required target size (256x256)
    img_resized = cv2.resize(img_np, target_size)

    # Convert to a 3D array (height, width, 1) as the model might expect 3D input
    img_resized = img_resized.reshape(target_size[0], target_size[1], 1)

    # Normalize the pixel values (same as dividing by 255)
    img_resized = img_resized / 255.0

    # Expand dimensions to match the batch size expected by the model
    img_array = np.expand_dims(img_resized, axis=0)  # Shape (1, 256, 256, 1)

    # Predict using the model
    predictions = model.predict(img_array)  # Get the prediction array
    predicted_class_idx = np.argmax(predictions, axis=1)[0]  # Get the index of the highest probability
    predicted_class = class_labels[predicted_class_idx]  # Map index to class label
    
    return predictions, predicted_class




def predict_brain(image,brain_model):
    size = (128, 128)
    image = ImageOps.fit(image, size, Image.LANCZOS)  # Use Image.LANCZOS instead of Image.ANTIALIAS
    image = image.convert('L')  # Ensure the image has 3 color channels (RGB)
    img = np.asarray(image)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    brain_prediction = brain_model.predict(img)
    return brain_prediction

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    model = load_lung_model()
    predictions,predicted_class = predict_lung( model,image)
    
    if predicted_class=='Benign case':
        st.write("Non cancerous but abnormal growth")
    elif predicted_class=='Malignant case':
        st.write("Cancerous growth")
    elif predicted_class=='Normal case':
        st.write("Non cancerous and normal growth") 
    
elif uploaded_file1 is not None:
    image = Image.open(uploaded_file1)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    model = load_brain_model()
    prediction = predict_brain(image, model)
    highest_value = np.max(prediction)
    
    optimal_threshold = 0.5   #Replace with the optimal threshold determined from evaluation
    if highest_value>=optimal_threshold:
        st.write("Positive for Brain Tumor")
    else:
        st.write("Negative for Brain Tumor")