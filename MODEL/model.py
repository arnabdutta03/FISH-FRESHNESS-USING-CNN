# model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    """
    Create a convolutional neural network (CNN) for classifying fish as fresh or non-fresh.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')  # Output layer for binary classification
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_images, train_labels):
    """
    Train the CNN model on the provided training data.
    """
    history = model.fit(train_images, train_labels, validation_split=0.2, epochs=10, batch_size=8)
    return history
