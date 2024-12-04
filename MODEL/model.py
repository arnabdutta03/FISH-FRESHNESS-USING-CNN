import tensorflow as tf
from tensorflow.keras import layers, models

# Define the Convolutional Neural Network (CNN)
def create_enhanced_model():
    """
    Creates and compiles an enhanced CNN model for fish freshness classification.
    
    Returns:
        model: Compiled enhanced CNN model.
    """
    model = models.Sequential([
        # First convolutional layer with 32 filters and ReLU activation
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.BatchNormalization(),  # Batch normalization
        layers.MaxPooling2D((2, 2)),  # Max pooling to reduce spatial dimensions
        
        # Second convolutional layer with 64 filters
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),  # Batch normalization
        layers.MaxPooling2D((2, 2)),  # Max pooling
        
        # Third convolutional layer with 128 filters
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),  # Batch normalization
        layers.MaxPooling2D((2, 2)),  # Max pooling
        
        # Fourth convolutional layer with 256 filters
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),  # Batch normalization
        layers.MaxPooling2D((2, 2)),  # Max pooling

        layers.Flatten(),  # Flatten feature maps into a single vector
        layers.Dense(512, activation='relu'),  # Fully connected dense layer
        layers.Dropout(0.5),  # Dropout layer for regularization
        layers.Dense(2, activation='softmax')  # Output layer for binary classification
    ])

    # Compile the model with Adam optimizer and sparse categorical crossentropy loss
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train the model on training data
def train_model(model, train_images, train_labels):
    """
    Trains the CNN model.
    
    Args:
        model: Compiled CNN model.
        train_images (ndarray): Array of training images.
        train_labels (ndarray): Array of training labels.
    
    Returns:
        history: Training history object containing metrics.
    """
    return model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2)
