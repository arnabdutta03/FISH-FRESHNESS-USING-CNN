import os
from data_loader import load_data, load_inference_data
from model import create_model, train_model
from evaluate_model import save_metrics, evaluate_model, classify_and_save_images
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf

# Constants for paths
FRESH_PATH = "D:\\FISH_SOM\\DATASET\\fresh"  # Path to fresh fish images
NON_FRESH_PATH = "D:\\FISH_SOM\\DATASET\\non-fresh"  # Path to non-fresh fish images
FISH_DATA_PATH = "D:\\FISH_SOM\\DATASET\\fish_data"  # Path to folder for inference images
SAVE_PATH = "D:\\FISH_SOM"  # Path to save results and model
MODEL_PATH = os.path.join(SAVE_PATH, "fish_model.h5")  # Path to save the trained model

def main():
    """
    Main pipeline for training, saving, evaluating, and classifying images with the fish freshness model.
    """
    # Load Data
    print("Loading data...")
    train_images, test_images, train_labels, test_labels = load_data(FRESH_PATH, NON_FRESH_PATH)
    print(f"Training samples: {len(train_images)}, Test samples: {len(test_images)}")
    
    # Create and Train the Model
    print("Creating and training the model...")
    model = create_model()  # Initialize the CNN model
    history = train_model(model, train_images, train_labels)  # Train the model
    print("Training complete.")
    
    # Save Training Metrics
    print("Saving training metrics...")
    save_metrics(history, SAVE_PATH)  # Save the training metrics as a JSON file
    
    # Save the Model
    print("Saving the model...")
    model.save(MODEL_PATH)  # Save the trained model
    print(f"Model saved to {MODEL_PATH}")
    
    # Evaluate the Model
    print("Evaluating the model...")
    evaluate_model(model, test_images, test_labels, SAVE_PATH)  # Evaluate the model on test data
    print("Evaluation complete.")
    
    # Classify Inference Images
    print("Classifying inference images...")
    image_paths = load_inference_data(FISH_DATA_PATH)  # Load image paths for inference
    
    # Load and preprocess images in batches for inference
    batch_size = 64  # Process 64 images at a time
    inference_images = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = [img_to_array(load_img(p, target_size=(128, 128))) / 255.0 for p in batch_paths]
        inference_images.append(np.array(batch_images))

    # Concatenate all batches into a single array
    inference_images = np.concatenate(inference_images, axis=0)
    classify_and_save_images(model, inference_images, image_paths, SAVE_PATH, batch_size=128)  # Classify and save images

    print("Classification and saving complete.")

# Execute the main function
if __name__ == "__main__":
    main()
