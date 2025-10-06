import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

from data_preprocessing import get_binary_dataset, get_multiclass_dataset
from model import build_cnn_model

def evaluate_single_model(test_loader, class_names, model_path, device, title):
    """
    Performs comprehensive evaluation of a single model and saves results.
    """
    print(f"\n--- Evaluation for: {title} ---")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}. Please train the model first.")
    
    num_classes = len(class_names)
    model = build_cnn_model(num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print(f"Model '{os.path.basename(model_path)}' loaded successfully.")
    
    y_true = []
    y_pred = []

    print("Generating predictions on the test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(np.unique(y_true)) < num_classes:
        print("\n" + "="*50 + "\n")
        print("Warning: The test set does not contain all classes. Skipping classification report and confusion matrix.")
        print(f"Expected classes: {class_names}")
        print(f"Classes found in test set: {np.unique(y_true)}")
        return

    print("\n" + "="*30 + " Classification Report " + "="*30 + "\n")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    os.makedirs("evaluation_results", exist_ok=True)
    report_path = f"evaluation_results/classification_report_{title.lower().replace(' ', '_')}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nClassification report saved to: {report_path}")

    print("\n" + "="*30 + " Confusion Matrix " + "="*30 + "\n")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap='Blues', xticks_rotation=90, ax=ax)
    plt.title(f"Confusion Matrix - {title}")
    confusion_matrix_path = f"evaluation_results/confusion_matrix_{title.lower().replace(' ', '_')}.png"
    plt.savefig(confusion_matrix_path, bbox_inches='tight')
    print(f"Confusion matrix saved to: {confusion_matrix_path}")
    plt.show()

def evaluate_model(device):
    """
    Main pipeline to evaluate both binary and multiclass models.
    """
    print("Starting model evaluation pipeline...")

    # Evaluate Binary Classifier
    _, _, test_loader_binary, binary_class_names = get_binary_dataset()
    evaluate_single_model(test_loader_binary, binary_class_names, "saved_models/binary_classifier.pth", device, "Binary Classification")
    
    print("\n" + "="*50 + "\n")

    # Evaluate Multiclass Classifier
    _, _, test_loader_multi, multiclass_names = get_multiclass_dataset()
    evaluate_single_model(test_loader_multi, multiclass_names, "saved_models/multiclass_classifier.pth", device, "Multiclass Classification")

    print("\nEvaluation pipeline complete.")

if __name__ == "__main__":
    from main import DEVICE
    evaluate_model(DEVICE)