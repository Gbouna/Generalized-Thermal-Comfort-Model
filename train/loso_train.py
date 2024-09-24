import os
import shutil
import logging
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks
from data_loader import create_temp_directories, prepare_loso_dataset
from model import get_cct_model
from typing import List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hyperparameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 16
num_epochs = 100
image_size = 224

# Ensure the checkpoint directory exists
os.makedirs('model_checkpoint_loso', exist_ok=True)

def delete_temp_directories(temp_dir: str) -> None:
    """
    Delete the temporary directories after use.
    Args:
        temp_dir (str): Path to the temporary directory.
    """
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logging.info(f"Temporary directory {temp_dir} deleted.")
        except Exception as e:
            logging.error(f"Failed to delete temp directory {temp_dir}. Error: {e}")

def prepare_datasets(train_dirs: List[str], val_dir: List[str], test_dir: str) -> Tuple:
    """
    Prepare training, validation, and test datasets using ImageDataGenerator.
    Args:
        train_dirs (List[str]): List of directories for training data.
        val_dir (List[str]): List of directories for validation data.
        test_dir (str): Directory for test data.
    Returns:
        Tuple: Training, validation, and test iterators along with temp directory.
    """
    logging.info(f"Preparing datasets... Train: {train_dirs}, Validation: {val_dir}, Test: {test_dir}")
    train_temp_dir, val_temp_dir, test_temp_dir = create_temp_directories(train_dirs, val_dir, test_dir)
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        width_shift_range=0.1,
        brightness_range=[0.2, 1.0],
        height_shift_range=0.1,
        horizontal_flip=True,
        rotation_range=45
    )
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    # Load datasets
    train_it = train_datagen.flow_from_directory(
        directory=train_temp_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    val_it = val_test_datagen.flow_from_directory(
        directory=val_temp_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    test_it = val_test_datagen.flow_from_directory(
        directory=test_temp_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    logging.info(f"Training images: {train_it.samples}, Validation images: {val_it.samples}, Test images: {test_it.samples}")
    return train_it, val_it, test_it, train_temp_dir

def plot_training_curves(history: tf.keras.callbacks.History, output_dir: str, identifier: str) -> None:
    """
    Plot and save training and validation accuracy and loss curves.
    Args:
        history (tf.keras.callbacks.History): Training history object.
        output_dir (str): Directory to save the plots.
        identifier (str): Identifier for the current experiment (e.g., model name).
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{identifier} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{identifier}_accuracy_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{identifier} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{identifier}_loss_curve.png"))
    plt.close()

def run_experiment(model: tf.keras.Model, train_it, val_it, test_it, checkpoint_dir: str, identifier: str) -> tf.keras.callbacks.History:
    """
    Run the training and evaluation pipeline for a given model.
    Args:
        model (tf.keras.Model): The model to train and evaluate.
        train_it: Training data iterator.
        val_it: Validation data iterator.
        test_it: Test data iterator.
        checkpoint_dir (str): Directory to save the model checkpoints.
        identifier (str): Identifier for the experiment (used in saving logs and models).
    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    checkpoint_filepath = os.path.join(checkpoint_dir, f"{identifier}_best_model.h5")
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                 tf.keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy")]
    )
    early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)
    model_checkpoint = callbacks.ModelCheckpoint(checkpoint_filepath, monitor="val_accuracy", save_best_only=True, save_weights_only=True)
    history = model.fit(train_it, validation_data=val_it, epochs=num_epochs, callbacks=[early_stopping, model_checkpoint], verbose=1)
    plot_training_curves(history, checkpoint_dir, identifier)
    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(test_it, verbose=1)
    logging.info(f"Test accuracy for {identifier}: {round(accuracy * 100, 2)}%")
    logging.info(f"Test top-5 accuracy for {identifier}: {round(top_5_accuracy * 100, 2)}%")

    # Generate predictions and confusion matrix
    predictions = model.predict(test_it, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_it.classes
    report = classification_report(y_true, y_pred, target_names=['Cool', 'Neutral', 'Warm'])
    with open(os.path.join(checkpoint_dir, f"{identifier}_classification_report.txt"), "w") as f:
        f.write(report)
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Save confusion matrices
    for cm_type, matrix, fmt in [("Normalized", cm_normalized, ".2%"), ("Non-Normalized", cm, "d")]:
        plt.figure(figsize=(10, 7))
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['Cool', 'Neutral', 'Warm'])
        disp.plot(cmap=plt.cm.Blues, values_format=fmt)
        plt.title(f'{identifier} Confusion Matrix ({cm_type})')
        plt.savefig(os.path.join(checkpoint_dir, f"{identifier}_confusion_matrix_{cm_type.lower()}.jpeg"))
        plt.close()
    return history

if __name__ == "__main__":
    base_dir = 'all_data'
    checkpoint_dir = "model_checkpoint_loso"
    # Prepare LOSO datasets
    loso_datasets = prepare_loso_dataset(base_dir)
    for group_index, (train_participants, val_participant, test_participant) in enumerate(loso_datasets, start=1):
        identifier = f"Group{group_index}_{test_participant}"
        train_data = [os.path.join(base_dir, participant, 'data') for participant in train_participants]
        val_data = [os.path.join(base_dir, participant, 'data') for participant in val_participant]
        test_data = os.path.join(base_dir, test_participant, 'data')
        train_it, val_it, test_it, temp_dir = prepare_datasets(train_data, val_data, test_data)
        model = get_cct_model()
        run_experiment(model, train_it, val_it, test_it, checkpoint_dir, identifier)
        delete_temp_directories(temp_dir)