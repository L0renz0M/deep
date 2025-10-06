import argparse
import tensorflow as tf
import os

# Import modules
from train import train_model
from evaluation import evaluate_model
from prediction import predict_image

# ==============================
# Global GPU Configuration (applied once at script start)
# ==============================
def global_gpu_setup():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Global GPU enabled with memory growth.")
        except RuntimeError as e:
            print(f"⚠️ Error enabling global memory growth: {e}")
    else:
        print("❌ No GPU detected. Operations will run on CPU.")
    tf.config.optimizer.set_jit(True) # Enable XLA for potential performance boost

# Run global GPU setup once at the very beginning
global_gpu_setup()

# ==============================
# MAIN SCRIPT
# ==============================
def main():
    parser = argparse.ArgumentParser(description="Manage CNN for Cassava leaf disease classification.")
    parser.add_argument('--train', action='store_true', help="Train the model.")
    parser.add_argument('--eval', action='store_true', help="Perform comprehensive evaluation of the model (report + confusion matrix).")
    parser.add_argument('--predict', type=str, help="Predict the class of a single image (specify image path).")

    args = parser.parse_args()

    # Use the appropriate functions based on arguments
    if args.train:
        print("\n--- Starting Training Process ---")
        train_model()
        print("--- Training Process Finished ---")
    elif args.eval:
        print("\n--- Starting Evaluation Process ---")
        evaluate_model()
        print("--- Evaluation Process Finished ---")
    elif args.predict:
        print("\n--- Starting Prediction Process ---")
        predict_image(args.predict)
        print("--- Prediction Process Finished ---")
    else:
        print("No operation specified. Use --train, --eval, or --predict.")
        parser.print_help()

if __name__ == "__main__":
    main()