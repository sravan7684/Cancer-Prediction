import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
import numpy as np

# Load the model
model = tf.keras.models.load_model('brain_tumor_model.h5')

# Data preparation
datagen = ImageDataGenerator(rescale=1./255)
validation_generator = datagen.flow_from_directory(
    'dataset2',  # Path to the validation dataset
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Get the ground truth labels
y_true = validation_generator.classes

# Make predictions
y_pred_prob = model.predict(validation_generator)
y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

# Calculate performance metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Plot ROC curve to determine the optimal threshold
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f'Optimal Threshold: {optimal_threshold}')
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