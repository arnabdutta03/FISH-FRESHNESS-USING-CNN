import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

# Load images from a given folder and assign a label to all images
def load_images_from_folder(folder, label):
    images, labels = [], []
    if not os.path.exists(folder):
        raise ValueError(f"Folder not found: {folder}")

    # Process each file in the folder
    with ThreadPoolExecutor(max_workers=8) as executor:  # Use 8 threads for parallel processing
        futures = []
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            futures.append(executor.submit(load_and_preprocess, file_path, label))

        for future in futures:
            img, lbl = future.result()
            if img is not None:
                images.append(img)
                labels.append(lbl)

    return images, labels

# Helper function to load and preprocess an image
def load_and_preprocess(file_path, label):
    try:
        img = load_img(file_path, target_size=(128, 128))
        img = img_to_array(img) / 255.0  # Normalize image to [0, 1]
        return img, label
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None, None

# Load fresh and non-fresh fish images for training and testing
def load_data(fresh_path, non_fresh_path):
    fresh_images, fresh_labels = load_images_from_folder(fresh_path, label=0)  # Fresh = 0
    non_fresh_images, non_fresh_labels = load_images_from_folder(non_fresh_path, label=1)  # Non-Fresh = 1

    # Combine data
    images = np.array(fresh_images + non_fresh_images)
    labels = np.array(fresh_labels + non_fresh_labels)

    # Shuffle and split data into training and testing sets
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    return train_images, test_images, train_labels, test_labels

# Load paths of images for inference from a specified folder
def load_inference_data(fish_data_path):
    image_paths = []

    if not os.path.exists(fish_data_path):
        raise ValueError(f"Folder not found: {fish_data_path}")

    # Retrieve image paths
    for subdir, dirs, files in os.walk(fish_data_path):
        for file in files:
            file_path = os.path.join(subdir, file)
            image_paths.append(file_path)

    return image_paths
