import torch
from torchvision import datasets, transforms
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Configurazioni di base
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
DATASET_DIR = "C:/Users/aloll/OneDrive/Desktop/data" 

# ===============================
# Dataset personalizzati
# ===============================
class RemappedDataset(torch.utils.data.Dataset):
    def __init__(self, full_dataset, remapping_dict, transform=None):
        self.full_dataset = full_dataset
        self.remapping_dict = remapping_dict
        self.transform = transform
        self.samples, self.targets = [], []

        for path, label in self.full_dataset.samples:
            class_name = self.full_dataset.classes[label]
            new_label = self.remapping_dict[class_name]
            self.samples.append((path, new_label))
            self.targets.append(new_label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = datasets.folder.default_loader(path)
        if self.transform:
            img = self.transform(img)
        return img, target


class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.targets = [s[1] for s in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = datasets.folder.default_loader(path)
        if self.transform:
            img = self.transform(img)
        return img, target


# ===============================
# Trasformazioni (augmentation)
# ===============================
def get_transforms(img_size):
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25)
        ]),
        "val_test": transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    }

# ===============================
# DATASET BINARIO
# ===============================
def get_binary_dataset(dataset_dir=DATASET_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    print(f"Loading binary dataset from: {dataset_dir}")
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(dataset_dir)

    transforms_dict = get_transforms(img_size)
    full_dataset = datasets.ImageFolder(root=dataset_dir)
    mapping = {name: 0 if "Healthy" in name else 1 for name in full_dataset.classes}
    remapped = RemappedDataset(full_dataset, mapping)

    # Split stratificato
    train_val, test, y_train_val, y_test = train_test_split(
        remapped.samples, remapped.targets, test_size=0.2, stratify=remapped.targets, random_state=SEED
    )
    train, val, y_train, y_val = train_test_split(
        train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=SEED
    )

    train_set = RemappedDataset(full_dataset, mapping, transform=transforms_dict["train"])
    val_set = RemappedDataset(full_dataset, mapping, transform=transforms_dict["val_test"])
    test_set = RemappedDataset(full_dataset, mapping, transform=transforms_dict["val_test"])

    train_set.samples, train_set.targets = train, y_train
    val_set.samples, val_set.targets = val, y_val
    test_set.samples, test_set.targets = test, y_test

    # Bilanciamento
    counts = np.bincount(train_set.targets)
    weights = 1. / counts
    sample_weights = [weights[t] for t in train_set.targets]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, ["Healthy", "Diseased"]


# ===============================
# DATASET MULTICLASSE
# ===============================
def get_multiclass_dataset(dataset_dir=DATASET_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    print(f"Loading multiclass dataset from: {dataset_dir}")
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

    # Bilanciamento
    counts = np.bincount(y_train)
    weights = 1. / counts
    sample_weights = [weights[t] for t in y_train]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, class_names
