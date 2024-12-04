import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report
import json
from concurrent.futures import ThreadPoolExecutor

# Save the training metrics (accuracy and loss) to a JSON file
def save_metrics(history, save_path):
    """
    Saves the training and validation metrics (accuracy and loss) to a JSON file.

    Args:
        history: History object containing training metrics.
        save_path (str): Directory path where metrics will be saved.
    """
    metrics_data = {
        'training_accuracy': history.history.get('accuracy', []),
        'validation_accuracy': history.history.get('val_accuracy', []),
        'training_loss': history.history.get('loss', []),
        'validation_loss': history.history.get('val_loss', [])
    }

    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    with open(os.path.join(save_path, 'training_metrics.json'), 'w') as f:
        json.dump(metrics_data, f)  # Save metrics as a JSON file
    print(f"Training metrics saved to {save_path}/training_metrics.json")

# Evaluate the model on test data and provide results
def evaluate_model(model, test_images, test_labels, save_path):
    """
    Evaluates the trained model on test data and saves the results to a file.

    Args:
        model: Trained model.
        test_images (ndarray): Array of test images.
        test_labels (ndarray): Array of test labels.
        save_path (str): Directory path where evaluation results will be saved.
    """
    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    test_accuracy_percentage = test_accuracy * 100

    # Generate predictions and calculate confusion matrix and classification report
    predictions = np.argmax(model.predict(test_images), axis=1)
    conf_matrix = confusion_matrix(test_labels, predictions)
    report = classification_report(
        test_labels,
        predictions,
        target_names=['Fresh', 'Non-Fresh'],
        digits=2
    )

    # Combine evaluation results into a single output
    output = (
        f"Confusion Matrix:\n{conf_matrix}\n\n"
        f"{report}\n"
        f"Test Loss: {test_loss:.4f}\n"
        f"Test Accuracy: {test_accuracy_percentage:.2f}%\n"
    )

    # Print and save the evaluation report
    os.makedirs(save_path, exist_ok=True)
    print(output)  # Display evaluation output
    with open(os.path.join(save_path, 'evaluation_report.txt'), 'w') as f:
        f.write(output)  # Save the report to a text file
    print(f"Evaluation report saved to {save_path}/evaluation_report.txt")

# Classify images and save them into corresponding folders: FRESH or NON_FRESH
def classify_and_save_images(model, images, image_paths, output_path, batch_size=128):
    """
    Classifies the given images using the model and saves them into FRESH or NON_FRESH folders.

    Args:
        model: Trained model.
        images (ndarray): Array of images to classify.
        image_paths (list): List of image file paths.
        output_path (str): Directory path to save the classified images.
        batch_size (int, optional): Number of images to process in a batch. Default is 128.
    """
    # Prepare output directories for classification results
    output_dir = os.path.join(output_path, 'OUTPUT')
    os.makedirs(output_dir, exist_ok=True)

    fresh_dir = os.path.join(output_dir, 'FRESH')
    non_fresh_dir = os.path.join(output_dir, 'NON_FRESH')
    os.makedirs(fresh_dir, exist_ok=True)
    os.makedirs(non_fresh_dir, exist_ok=True)

    num_images = len(images)  # Total number of images to classify

    # Use ThreadPoolExecutor for concurrent file operations
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in range(0, num_images, batch_size):
            # Process images in batches
            batch_images = images[i:i+batch_size]
            batch_paths = image_paths[i:i+batch_size]
            
            # Perform batch predictions
            predictions = np.argmax(model.predict(np.array(batch_images)), axis=1)

            # Concurrently save images to the appropriate folder (FRESH or NON_FRESH)
            futures = []
            for pred, img_path in zip(predictions, batch_paths):
                destination = fresh_dir if pred == 0 else non_fresh_dir
                futures.append(executor.submit(shutil.copy, img_path, destination))

            # Wait for all file operations to complete
            for future in futures:
                future.result()

    print(f"Classified images saved to {output_dir}")
