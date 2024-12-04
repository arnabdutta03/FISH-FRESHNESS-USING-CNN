# data_loader.py
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_data(fresh_path, non_fresh_path):
    """
    Load the fresh and non-fresh fish images for training.
    """
    def load_images_from_folder(folder, label):
        images, labels = [], []
        for filename in os.listdir(folder):
            img = load_img(os.path.join(folder, filename), target_size=(128, 128))
            img = img_to_array(img) / 255.0
            images.append(img)
            labels.append(label)
        return images, labels

    # Load Fresh and Non-Fresh Data
    fresh_images, fresh_labels = load_images_from_folder(fresh_path, label=0)  # Fresh = 0
    non_fresh_images, non_fresh_labels = load_images_from_folder(non_fresh_path, label=1)  # Non-Fresh = 1

    # Combine Data
    images = np.array(fresh_images + non_fresh_images)
    labels = np.array(fresh_labels + non_fresh_labels)

    # Shuffle and Split Data
    from sklearn.model_selection import train_test_split
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    
    return train_images, test_images, train_labels, test_labels

def load_inference_data(fish_data_path):
    """
    Load images for inference from the fish_data folder.
    """
    images = []
    image_paths = []

    # Traverse the fish_data folder
    for subdir, dirs, files in os.walk(fish_data_path):
        for file in files:
            img_path = os.path.join(subdir, file)
            img = load_img(img_path, target_size=(128, 128))
            img = img_to_array(img) / 255.0
            images.append(img)
            image_paths.append(img_path)

    return np.array(images), image_paths
