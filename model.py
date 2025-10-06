import torch
import torch.nn as nn
from torchvision import models

def build_cnn_model(num_classes):
    """
    Builds a pre-trained ResNet18 model for classification.
    """
    print(f"Building pre-trained ResNet18 model for {num_classes} classes...")
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # Freeze all layers except the final classification layer
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    return model