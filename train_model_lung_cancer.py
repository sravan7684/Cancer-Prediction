import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Data preparation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    'cancer_dataset',  # Path to the dataset
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',  # Change to 'categorical' for multi-class classification
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'cancer_dataset',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',  # Change to 'categorical' for multi-class classification
    subset='validation'
)

# Build the model
model = Sequential([
    Input(shape=(150, 150, 3)),  # Use Input layer as the first layer
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Change to 3 units for 3 classes and use 'softmax' activation
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('lung_cancer_model.keras', save_best_only=True)  # Update file path to end with .keras

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation Accuracy: {accuracy}')