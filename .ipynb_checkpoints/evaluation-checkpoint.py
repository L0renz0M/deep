import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
#evaluation
# Import modules
from data_preprocessing import get_dataset, IMG_SIZE
from model import build_cnn_model # Import for custom objects if you re-introduce them

def setup_gpu():
    """Configures GPU memory growth."""
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU enabled with memory growth.")
        except RuntimeError as e:
            print(f"⚠️ Error enabling memory growth: {e}")
    else:
        print("❌ No GPU detected. Running evaluation on CPU.")
    tf.config.optimizer.set_jit(True)

def evaluate_model(model_path="saved_models/best_model.keras"):
    """
    Performs comprehensive evaluation of the model.
    """
    setup_gpu()
    print("Starting model evaluation...")

    # Load dataset (only test_ds is needed for evaluation)
    with tf.device('/CPU:0'): # Data loading typically faster on CPU
        _, _, test_ds, class_names = get_dataset()

    # Load the trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}. Please train the model first.")

    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        # If you add custom layers (like ChannelAttention, SpatialAttention),
        # they must be provided in custom_objects when loading.
        # Example: custom_objects={"ChannelAttention": ChannelAttention, "SpatialAttention": SpatialAttention}
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting to load without custom objects (if they were not part of the base keras layers).")
            model = tf.keras.models.load_model(model_path, compile=False) # Try loading without compilation and then recompile if needed
            # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            # Note: If custom layers are truly part of the model's structure,
            # this loading will fail unless their classes are defined and passed.
            # I've commented out your attention layers for now, so this should work.

    print(f"Model '{os.path.basename(model_path)}' loaded successfully.")
    model.summary()

    # Predict on the test dataset
    y_true = []
    y_pred = []

    print("Generating predictions on the test set...")
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())

    # Convert to numpy arrays for sklearn
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Print and Save Classification Report
    print("\n" + "="*30 + " Classification Report " + "="*30 + "\n")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    os.makedirs("evaluation_results", exist_ok=True)
    report_path = "evaluation_results/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nClassification report saved to: {report_path}")

    # Plot and Save Confusion Matrix
    print("\n" + "="*30 + " Confusion Matrix " + "="*30 + "\n")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10)) # Adjust figure size for better readability
    disp.plot(cmap='Blues', xticks_rotation=90, ax=ax) # Rotate x-axis labels
    plt.title("Confusion Matrix")
    confusion_matrix_path = "evaluation_results/confusion_matrix.png"
    plt.savefig(confusion_matrix_path, bbox_inches='tight')
    print(f"Confusion matrix saved to: {confusion_matrix_path}")
    plt.show()
    print("\nEvaluation complete.")

if __name__ == "__main__":
    evaluate_model()