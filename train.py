import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time
import logging
from datetime import datetime
import sys

from data_preprocessing import get_binary_dataset, get_multiclass_dataset, get_transforms, FilteredDataset
from model import build_cnn_model
from torchvision import datasets
from sklearn.model_selection import train_test_split

# ==========================================
# Configurazione globale
# ==========================================
DATASET_DIR = "C:/Users/aloll/OneDrive/Desktop/data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 12
PATIENCE = 8

# ==========================================
# Fix encoding Windows per logging UTF-8
# ==========================================
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ==========================================
# Logging
# ==========================================
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# Loss con label smoothing
# ==========================================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / self.num_classes)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=self.dim))


# ==========================================
# Mixup augmentation
# ==========================================
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ==========================================
# Training di un singolo modello
# ==========================================
def train_single_model(train_loader, val_loader, num_classes, model_path, device):
    model = build_cnn_model(num_classes).to(device)
    criterion = LabelSmoothingLoss(num_classes, smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_val_loss = float("inf")
    patience_counter = 0

    logger.info(f"Starting training for {num_classes}-class model.")
    logger.info(f"Model checkpoint path: {model_path}")

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader, start=1):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, y_a, y_b, lam = mixup_data(inputs, labels, alpha=0.4)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                logger.info(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_loader)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        duration = time.time() - start_time
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] finished in {duration:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.6f}")

        # Early stopping & checkpoint
        if val_loss < best_val_loss - 1e-3:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            logger.info("Model improved and saved ✅")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                logger.info("Early stopping triggered ⏹")
                break

        logger.info("-" * 60)

    logger.info("Training completed 🎯")


# ==========================================
# Funzione principale di training
# ==========================================
def train_model(device):
    # Binary classifier
    logger.info("--- Training Binary Classifier ---")
    train_loader, val_loader, _, class_names = get_binary_dataset()
    train_single_model(train_loader, val_loader, len(class_names), "saved_models/binary_classifier.pth", device)

    logger.info("=" * 70)

    # Multiclass classifier
    logger.info("--- Training Multiclass Classifier ---")
    train_loader, val_loader, _, class_names = get_multiclass_dataset()
    train_single_model(train_loader, val_loader, len(class_names), "saved_models/multiclass_classifier.pth", device)


# ==========================================
# Funzione per il dataset multiclasse
# ==========================================
def get_multiclass_dataset(dataset_dir=DATASET_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    logger.info(f"Loading multiclass dataset from: {dataset_dir}")
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(dataset_dir)

    transforms_dict = get_transforms(img_size)
    full_dataset = datasets.ImageFolder(root=dataset_dir)
    class_names = full_dataset.classes
    targets = np.array(full_dataset.targets)

    # Split stratificato
    train_val, test, y_train_val, y_test = train_test_split(
        full_dataset.samples, targets, test_size=0.2, stratify=targets, random_state=SEED
    )
    train, val, y_train, y_val = train_test_split(
        train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=SEED
    )

    train_set = FilteredDataset(train, transform=transforms_dict["train"])
    val_set = FilteredDataset(val, transform=transforms_dict["val_test"])
    test_set = FilteredDataset(test, transform=transforms_dict["val_test"])

    # Weighted sampler evitando divisione per zero
    counts = np.bincount(y_train)
    weights = np.array([1.0 / c if c > 0 else 0.0 for c in counts])
    sample_weights = [weights[t] for t in y_train]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, class_names
