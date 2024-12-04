import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report
import json
from concurrent.futures import ThreadPoolExecutor

# Save the training metrics (accuracy and loss) to a JSON file
def save_metrics(history, save_path):
    metrics_data = {
        'training_accuracy': history.history.get('accuracy', []),
        'validation_accuracy': history.history.get('val_accuracy', []),
        'training_loss': history.history.get('loss', []),
        'validation_loss': history.history.get('val_loss', [])
    }
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'training_metrics.json'), 'w') as f:
        json.dump(metrics_data, f)
    print(f"Training metrics saved to {save_path}/training_metrics.json")

# Evaluate the model on test data and provide results
def evaluate_model(model, test_images, test_labels, save_path):
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    test_accuracy_percentage = test_accuracy * 100

    # Generate predictions
    predictions = np.argmax(model.predict(test_images), axis=1)
    conf_matrix = confusion_matrix(test_labels, predictions)
    report = classification_report(
        test_labels,
        predictions,
        target_names=['Fresh', 'Non-Fresh'],
        digits=2
    )

    # Combine results into a single output
    output = (
        f"Confusion Matrix:\n{conf_matrix}\n\n"
        f"{report}\n"
        f"Test Loss: {test_loss:.4f}\n"
        f"Test Accuracy: {test_accuracy_percentage:.2f}%\n"
    )

    # Print and save the evaluation report
    os.makedirs(save_path, exist_ok=True)
    print(output)
    with open(os.path.join(save_path, 'evaluation_report.txt'), 'w') as f:
        f.write(output)
    print(f"Evaluation report saved to {save_path}/evaluation_report.txt")

# Classify images and save them into corresponding folders: FRESH or NON_FRESH
def classify_and_save_images(model, images, image_paths, output_path):
    # Prepare output directories
    output_dir = os.path.join(output_path, 'OUTPUT')
    os.makedirs(output_dir, exist_ok=True)

    fresh_dir = os.path.join(output_dir, 'FRESH')
    non_fresh_dir = os.path.join(output_dir, 'NON_FRESH')
    os.makedirs(fresh_dir, exist_ok=True)
    os.makedirs(non_fresh_dir, exist_ok=True)

    # Use ThreadPoolExecutor for concurrent classification and saving
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for img_array, img_path in zip(images, image_paths):
            futures.append(executor.submit(classify_and_save, model, img_array, img_path, fresh_dir, non_fresh_dir))

        for future in futures:
            future.result()

    print(f"Classified images saved to {output_dir}")

# Helper function for classifying and saving an image
def classify_and_save(model, img_array, img_path, fresh_dir, non_fresh_dir):
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = np.argmax(model.predict(img_array), axis=1)

    # Save the image to the corresponding folder based on prediction
    destination = fresh_dir if prediction == 0 else non_fresh_dir
    shutil.copy(img_path, destination)
