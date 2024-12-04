# main.py
import os
from data_loader import load_data, load_inference_data
from model import create_model, train_model
from evaluate_model import save_metrics, evaluate_model, classify_and_save_images

# Data Paths
FRESH_PATH = "D:\\FISH_SOM\\DATASET\\fresh"
NON_FRESH_PATH = "D:\\FISH_SOM\\DATASET\\non-fresh"
FISH_DATA_PATH = "D:\\FISH_SOM\\DATASET\\fish_data"
SAVE_PATH = "D:\\FISH_SOM"

# Load Data for Training
train_images, test_images, train_labels, test_labels = load_data(FRESH_PATH, NON_FRESH_PATH)

# Create and Train Model
model = create_model()
history = train_model(model, train_images, train_labels)

# Save Training Metrics
save_metrics(history, SAVE_PATH)

# Save the model weights
model.save(os.path.join(SAVE_PATH, "fish_model.h5"))  # Save the model using Keras' save method

# Evaluate the model
evaluate_model(model, test_images, test_labels, SAVE_PATH)

# Load the inference data (fish_data folder with many images)
inference_images, image_paths = load_inference_data(FISH_DATA_PATH)

# Classify and save images into FRESH and NON_FRESH folders
classify_and_save_images(model, inference_images, image_paths, SAVE_PATH)
