# evaluate_model.py
import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img
from sklearn.metrics import confusion_matrix, classification_report

def save_metrics(history, save_path):
    """
    Save the training metrics (accuracy and loss) to a file.
    """
    import json
    accuracy_data = {
        'training_accuracy': history.history['accuracy'],
        'validation_accuracy': history.history['val_accuracy'],
        'training_loss': history.history['loss'],
        'validation_loss': history.history['val_loss'],
    }
    with open(os.path.join(save_path, 'training_metrics.json'), 'w') as f:
        json.dump(accuracy_data, f)

def evaluate_model(model, test_images, test_labels, save_path):
    """
    Evaluate the model on test data and provide combined output for test accuracy,
    confusion matrix, and classification report in percentage format.
    """
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    test_accuracy_percentage = test_accuracy * 100

    # Predictions and results
    predictions = np.argmax(model.predict(test_images), axis=1)
    conf_matrix = confusion_matrix(test_labels, predictions)
    report = classification_report(
        test_labels,
        predictions,
        target_names=['Fresh', 'Non-Fresh'],
        digits=2  # Ensures percentage-style output
    )

    # Combine everything into a single formatted output
    output = (
        f"Confusion Matrix:\n{conf_matrix}\n\n"
        f"{report}\n"
        f"Test Accuracy: {test_accuracy_percentage:.2f}%"
    )

    # Print and save the combined output
    print(output)
    with open(os.path.join(save_path, 'evaluation_report.txt'), 'w') as f:
        f.write(output)

def classify_and_save_images(model, images, image_paths, output_path):
    """
    Classify images and save them into the corresponding folder (FRESH or NON_FRESH).
    """
    # Create OUTPUT directory if it doesn't exist
    output_dir = os.path.join(output_path, 'OUTPUT')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create FRESH and NON_FRESH folders
    fresh_dir = os.path.join(output_dir, 'FRESH')
    non_fresh_dir = os.path.join(output_dir, 'NON_FRESH')
    os.makedirs(fresh_dir, exist_ok=True)
    os.makedirs(non_fresh_dir, exist_ok=True)

    # Classify images and move them to respective folders
    predictions = np.argmax(model.predict(images), axis=1)

    for idx, img_path in enumerate(image_paths):
        if predictions[idx] == 0:  # Fresh fish
            shutil.copy(img_path, os.path.join(fresh_dir, os.path.basename(img_path)))
        else:  # Non-Fresh fish
            shutil.copy(img_path, os.path.join(non_fresh_dir, os.path.basename(img_path)))
