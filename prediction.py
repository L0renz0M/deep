import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F

from data_preprocessing import IMG_SIZE, DATASET_DIR
from model import build_cnn_model

def load_and_preprocess_image(image_path):
    """
    Loads an image, applies transformations, and prepares it for a PyTorch model.
    """
    print(f"Loading and preprocessing image: {image_path}")
    
    preprocess = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0) 
    
    return img_tensor

def get_class_names(dataset_dir=DATASET_DIR, binary_mode=True):
    """
    Infers class names from the dataset directory structure.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}. Cannot infer class names.")
    
    class_names = sorted([d.name for d in os.scandir(dataset_dir) if d.is_dir()])
    
    if binary_mode:
        return ["Healthy", "Diseased"]
    else:
        return [name for name in class_names if "Healthy" not in name]

def predict_image(image_path, device):
    """
    Performs a two-step prediction: binary (healthy/diseased) then multiclass if diseased.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    # Step 1: Binary Classification
    print("\n--- Performing Binary Classification ---")
    binary_model_path = "saved_models/binary_classifier.pth"
    if not os.path.exists(binary_model_path):
        print(f"Error: Binary model not found at {binary_model_path}. Please train the model first.")
        return

    binary_class_names = get_class_names(binary_mode=True)
    num_classes_binary = len(binary_class_names)
    binary_model = build_cnn_model(num_classes_binary).to(device)
    binary_model.load_state_dict(torch.load(binary_model_path))
    binary_model.eval()

    img_tensor = load_and_preprocess_image(image_path).to(device)

    with torch.no_grad():
        binary_outputs = binary_model(img_tensor)
    
    binary_probabilities = F.softmax(binary_outputs, dim=1)[0]
    predicted_binary_idx = torch.argmax(binary_probabilities).item()
    predicted_binary_name = binary_class_names[predicted_binary_idx]
    binary_confidence = binary_probabilities[predicted_binary_idx].item() * 100

    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Binary Prediction: {predicted_binary_name} with {binary_confidence:.2f}% confidence.")
    
    # Step 2: Multiclass Classification (if diseased)
    if predicted_binary_name == "Diseased":
        print("\n--- Performing Multiclass Classification ---")
        multiclass_model_path = "saved_models/multiclass_classifier.pth"
        if not os.path.exists(multiclass_model_path):
            print(f"Error: Multiclass model not found at {multiclass_model_path}. Please train the model first.")
            return

        multiclass_class_names = get_class_names(binary_mode=False)
        num_classes_multiclass = len(multiclass_class_names)
        multiclass_model = build_cnn_model(num_classes_multiclass).to(device)
        multiclass_model.load_state_dict(torch.load(multiclass_model_path))
        multiclass_model.eval()

        with torch.no_grad():
            multiclass_outputs = multiclass_model(img_tensor)
        
        multiclass_probabilities = F.softmax(multiclass_outputs, dim=1)[0]
        predicted_multiclass_idx = torch.argmax(multiclass_probabilities).item()
        predicted_multiclass_name = multiclass_class_names[predicted_multiclass_idx]
        multiclass_confidence = multiclass_probabilities[predicted_multiclass_idx].item() * 100

        print(f"Specific Disease Prediction: {predicted_multiclass_name} with {multiclass_confidence:.2f}% confidence.")

        print("\nAll disease probabilities:")
        for i, prob in enumerate(multiclass_probabilities):
            print(f"  {multiclass_class_names[i]}: {prob.item():.2%}")
    else:
        print("Image classified as Healthy, no further classification needed.")