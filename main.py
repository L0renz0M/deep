import argparse
import torch
import os
from train import train_model
from evaluation import evaluate_model
from prediction import predict_image


def global_gpu_setup():
  
    if torch.cuda.is_available():
        
        device = torch.device("cuda")
    else:
        
        device = torch.device("cpu")
    return device


DEVICE = global_gpu_setup()


def main():
    parser = argparse.ArgumentParser(description="Manage CNN for Cassava leaf disease classification.")
    parser.add_argument('--train', action='store_true', help="Train the models.")
    parser.add_argument('--eval', action='store_true', help="Perform comprehensive evaluation of the models.")
    parser.add_argument('--predict', type=str, help="Predict the class of a single image (specify image path).")

    args = parser.parse_args()

    
    if args.train:
        print("\n--- Starting Training Process ---")
        train_model(DEVICE)
        print("--- Training Process Finished ---")
    elif args.eval:
        print("\n--- Starting Evaluation Process ---")
        evaluate_model(DEVICE)
        print("--- Evaluation Process Finished ---")
    elif args.predict:
        print("\n--- Starting Prediction Process ---")
        predict_image(args.predict, DEVICE)
        print("--- Prediction Process Finished ---")
    else:
        print("No operation specified. Use --train, --eval, or --predict.")
        parser.print_help()

if __name__ == "__main__":
    main()