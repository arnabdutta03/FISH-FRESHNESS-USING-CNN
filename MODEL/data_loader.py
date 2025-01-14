import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

# Load images from a given folder and assign a label to all images
def load_images_from_folder(folder, label):
    """
    Loads and preprocesses images from the specified folder and assigns a label to each image.

    Args:
        folder (str): Path to the folder containing images.
        label (int): Label assigned to all images in the folder.

    Returns:
        images (list): Preprocessed image data.
        labels (list): Corresponding labels for the images.
    """
    images, labels = [], []
    if not os.path.exists(folder):
        raise ValueError(f"Folder not found: {folder}")

    # Use ThreadPoolExecutor to process multiple files concurrently
    with ThreadPoolExecutor(max_workers=8) as executor:
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
    """
    Loads and preprocesses a single image.

    Args:
        file_path (str): Path to the image file.
        label (int): Label assigned to the image.

    Returns:
        img (ndarray): Preprocessed image array.
        label (int): Label of the image.
    """
    try:
        # Resize the image to (128, 128) and normalize pixel values
        img = load_img(file_path, target_size=(128, 128))
        img = img_to_array(img) / 255.0
        return img, label
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None, None

# Load data from fresh and non-fresh image folders
def load_data(fresh_path, non_fresh_path):
    """
    Loads images and labels from fresh and non-fresh fish folders.

    Args:
        fresh_path (str): Path to fresh fish images.
        non_fresh_path (str): Path to non-fresh fish images.

    Returns:
        train_images, test_images, train_labels, test_labels: Dataset split into training and testing sets.
    """
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

# Load paths of images for inference
def load_inference_data(fish_data_path):
    """
    Retrieves paths of all images in the specified folder for inference.

    Args:
        fish_data_path (str): Path to the folder containing inference images.

    Returns:
        image_paths (list): List of paths to all images.
    """
    image_paths = []

    if not os.path.exists(fish_data_path):
        raise ValueError(f"Folder not found: {fish_data_path}")

    for subdir, _, files in os.walk(fish_data_path):
        for file in files:
            image_paths.append(os.path.join(subdir, file))

    return image_paths
