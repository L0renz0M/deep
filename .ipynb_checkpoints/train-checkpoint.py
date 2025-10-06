import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os

# Import modules
from data_preprocessing import get_dataset, IMG_SIZE
from model import build_cnn_model
from callbacks import get_callbacks

EPOCHS = 50 # Increased epochs, EarlyStopping will stop it early if needed

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
        print("❌ No GPU detected. Training on CPU.")
    tf.config.optimizer.set_jit(True) # Enable XLA (Accelerated Linear Algebra)

def train_model():
    """
    Main function to load data, build model, and train.
    """
    setup_gpu()
    print("Starting the training process...")

    # 1. Load and Preprocess Data
    with tf.device('/CPU:0'): # Data loading typically faster on CPU
        train_ds, val_ds, test_ds, class_names = get_dataset()
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")

    # 2. Calculate Class Weights (Important for imbalanced datasets)
    # Collect all labels from the training dataset
    print("Calculating class weights...")
    labels = []
    for _, batch_labels in train_ds.unbatch().as_numpy_iterator():
        labels.append(batch_labels)
    y_train_flat = np.array(labels)

    if len(np.unique(y_train_flat)) < num_classes:
        print("Warning: Not all classes present in training set for class weight calculation. This might be due to small dataset size or dataset split.")

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_flat),
        y=y_train_flat
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weights_dict}")

    # 3. Build the Model
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        model = build_cnn_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        model.summary()

    # 4. Define Callbacks
    os.makedirs("saved_models", exist_ok=True)
    callbacks = get_callbacks(model_save_path="saved_models/best_model.keras")

    # 5. Train the Model
    print("Fitting the model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights_dict
    )

    # Load the best model saved by ModelCheckpoint for evaluation/final save
    if os.path.exists("saved_models/best_model.keras"):
        final_model = tf.keras.models.load_model("saved_models/best_model.keras")
        print("\nLoaded best model for final saving and evaluation.")
    else:
        final_model = model
        print("\nBest model not found, using the last trained model.")

    final_model.save("saved_models/cassava_cnn_model_final.keras")
    print("Final model saved successfully as 'cassava_cnn_model_final.keras'.")

    # You can return history or other objects if needed for visualization in main.py
    return history, final_model, test_ds, class_names

if __name__ == "__main__":
    train_model()