import tensorflow as tf
import numpy as np
import os
from PIL import Image # For image loading
from data_preprocessing import IMG_SIZE, DATASET_DIR # Reuse constants

def setup_gpu_for_prediction():
    """Configures GPU memory growth for prediction."""
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU enabled with memory growth for prediction.")
        except RuntimeError as e:
            print(f"⚠️ Error enabling memory growth: {e}")
    else:
        print("❌ No GPU detected. Running prediction on CPU.")
    tf.config.optimizer.set_jit(True)

def load_and_preprocess_image(image_path):
    """
    Loads an image, resizes it, and normalizes it for prediction.
    """
    print(f"Loading and preprocessing image: {image_path}")
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

def get_class_names(dataset_dir=DATASET_DIR):
    """
    Infers class names from the dataset directory structure.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}. Cannot infer class names.")
    # A simple way to get class names by listing subdirectories
    class_names = sorted([d.name for d in os.scandir(dataset_dir) if d.is_dir()])
    if not class_names:
        print(f"Warning: No subdirectories found in {dataset_dir}. Class names might be incorrect.")
    return class_names

def predict_image(image_path, model_path="saved_models/best_model.keras"):
    """
    Loads a trained model and predicts the class of a single image.
    """
    setup_gpu_for_prediction()

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        return

    # Load the model
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Model '{os.path.basename(model_path)}' loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting to load without custom objects (if they were not part of the base keras layers).")
            model = tf.keras.models.load_model(model_path, compile=False)
            # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    # Preprocess the image
    img_array = load_and_preprocess_image(image_path)

    # Get class names
    class_names = get_class_names()
    if not class_names:
        print("Could not infer class names from dataset directory. Prediction labels might be generic.")
        # Provide dummy names if inference fails
        class_names = [f"Class {i}" for i in range(model.output_shape[-1])]

    # Make prediction
    print("Making prediction...")
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0])

    predicted_class_idx = np.argmax(score)
    predicted_class_name = class_names[predicted_class_idx]
    confidence = np.max(score) * 100

    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Predicted class: {predicted_class_name}")
    print(f"Confidence: {confidence:.2f}%")

    print("\nAll probabilities:")
    for i, prob in enumerate(score):
        print(f"  {class_names[i]}: {prob.numpy():.2%}")

if __name__ == "__main__":
    # Example usage: replace 'path/to/your/image.jpg' with a real image path
    # Make sure you have a trained model at 'saved_models/best_model.keras'
    # and a 'dataset' directory structure for class names inference.
    # predict_image("C:\\Users\\andre\\Desktop\\CNNModel\\test_image.jpg")
    print("Run this script from main.py using --predict IMAGE_PATH")