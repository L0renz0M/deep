import torch
import torch.nn as nn
from torchvision import models

def build_cnn_model(num_classes: int):
    """
    Costruisce un modello ResNet18 pre-addestrato con fine-tuning parziale.
    Include BatchNorm e Dropout per ridurre l'overfitting.
    """
    print(f"Building fine-tuned ResNet18 model for {num_classes} classes...")
    model = models.resnet18(weights='IMAGENET1K_V1')

    # Sblocca solo i layer finali per il fine-tuning
    for name, param in model.named_parameters():
        if "layer4" in name or "layer3" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Nuova testa di classificazione
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    return model
