import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import time
import os

class TrainingProgressLogger(Callback):
    """
    A Keras callback to log training progress including epoch duration and ETA.
    """
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.epoch_times = []
        print("\nStarting training...\n")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start_time
        self.epoch_times.append(elapsed)
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.params['epochs'] - (epoch + 1)
        eta = avg_epoch_time * remaining_epochs

        # Format ETA for better readability
        eta_str = f"{eta:.2f}s"
        if eta > 60:
            eta_str = f"{eta/60:.2f} min"
        if eta > 3600:
            eta_str = f"{eta/3600:.2f} hours"

        print(f"Epoch {epoch + 1}/{self.params['epochs']} completed - "
              f"Duration: {elapsed:.2f}s - ETA: {eta_str}")

def get_callbacks(model_save_path="saved_models/best_model.keras"):
    """
    Returns a list of Keras callbacks for training.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8, # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss', # Monitor validation loss for saving best model
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5, # Reduce LR by 50%
            patience=4, # Increased patience for LR reduction
            min_lr=1e-7, # Ensure learning rate doesn't go too low
            verbose=1
        ),
        TrainingProgressLogger(),
        tf.keras.callbacks.CSVLogger('training_log.csv') # Log metrics to a CSV file
    ]
    return callbacks

if __name__ == '__main__':
    # This block is just for demonstration if you want to test the callback
    print("Testing callback setup (no actual training).")
    os.makedirs("saved_models", exist_ok=True)
    _ = get_callbacks()
    print("Callbacks instantiated.")