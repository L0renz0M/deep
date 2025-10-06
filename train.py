import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import torch.nn.functional as F # Necessario per log_softmax

from data_preprocessing import get_binary_dataset, get_multiclass_dataset
from model import build_cnn_model

EPOCHS = 50
PATIENCE = 8

# =========================================================
# === CLASSE CUSTOM PER IL LABEL SMOOTHING (IL TUO SMOOTHEN) ===
# =========================================================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        # 1. Confidence è il peso assegnato alla classe vera (es. 0.9)
        self.confidence = 1.0 - smoothing
        # 2. Smoothing è la probabilità distribuita (es. 0.1)
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim

    def forward(self, pred, target):
        # pred sono i logit del modello (output non normalizzato)
        
        # 1. Trasforma i logit in log-probabilità
        log_probs = F.log_softmax(pred, dim=self.dim)
        
        # 2. Costruisci la distribuzione smoothed del target
        with torch.no_grad():
            # Inizializza la distribuzione target (true_dist) con il termine di smoothing (epsilon / K)
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / self.num_classes)
            
            # Assegna il termine di confidence (1 - epsilon) alla posizione della classe vera
            # target.data.unsqueeze(1) trasforma (batch_size,) in (batch_size, 1) per l'indicizzazione
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        # 3. Calcola la Loss (equivalente alla Cross-Entropy su distribuzioni arbitrarie)
        # Loss = - sum(smoothed_target * log_prob)
        loss = torch.sum(-true_dist * log_probs, dim=self.dim)
        
        # Restituisce la media della loss sul batch
        return torch.mean(loss)


def train_single_model(train_loader, val_loader, num_classes, model_path, device):
    """
    Trains a single model instance and saves the best model.
    """
    model = build_cnn_model(num_classes).to(device)
    
    # === SOSTITUZIONE: Usa LabelSmoothingLoss invece di nn.CrossEntropyLoss() ===
    # Imposta un smoothing di 0.1. Puoi provare 0.15 o 0.2.
    criterion = LabelSmoothingLoss(num_classes=num_classes, smoothing=0.1) 
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5, min_lr=1e-7)

    best_val_loss = float('inf')
    patience_counter = 0

    print("Fitting the model...")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # La LabelSmoothingLoss prende i logit (outputs) e le etichette (labels)
            loss = criterion(outputs, labels) 
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss/len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {epoch_loss:.4f}")

        # VALIDATION PHASE
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # Usiamo la LabelSmoothingLoss anche per la validazione per coerenza di monitoraggio.
                val_loss += criterion(outputs, labels).item() 
                
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print("Model saved! Best validation loss so far.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break
        
        end_time = time.time()
        print(f"Epoch {epoch+1}/{EPOCHS} completed in {end_time - start_time:.2f}s")
        print("-" * 30)

    print("Training finished.")


def train_model(device):
    """
    Main pipeline to train both binary and multiclass models.
    """
    print("--- Starting Binary Classification Training ---")
    train_loader_binary, val_loader_binary, _, binary_class_names = get_binary_dataset()
    # num_classes = 2.
    train_single_model(train_loader_binary, val_loader_binary, len(binary_class_names), "saved_models/binary_classifier.pth", device)
    
    print("\n" + "="*50 + "\n")
    
    print("--- Starting Multiclass Classification Training (Diseased Leaves) ---")
    train_loader_multi, val_loader_multi, _, multiclass_names = get_multiclass_dataset()
    
    train_single_model(train_loader_multi, val_loader_multi, len(multiclass_names), "saved_models/multiclass_classifier.pth", device)