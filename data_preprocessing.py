import torch
from torchvision import datasets, transforms
import os
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
DATASET_DIR = "C:\\Users\\andre\\Desktop\\CNNModel\\dataset" # Change the path

# Define custom dataset classes at the global level
class RemappedDataset(torch.utils.data.Dataset):
    def __init__(self, full_dataset, remapping_dict, transform=None):
        self.full_dataset = full_dataset
        self.remapping_dict = remapping_dict
        self.transform = transform
        
        self.samples = []
        self.targets = []
        
        for path, label in self.full_dataset.samples:
            original_class_name = self.full_dataset.classes[label]
            remapped_label = self.remapping_dict[original_class_name]
            self.samples.append((path, remapped_label))
            self.targets.append(remapped_label)

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

def get_binary_dataset(dataset_dir=DATASET_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Loads dataset for binary classification (healthy vs. diseased)
    """
    print(f"Loading binary dataset from: {dataset_dir}")
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    binary_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val_test': transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    full_dataset = datasets.ImageFolder(root=dataset_dir)
    class_names = full_dataset.classes
    binary_mapping = {name: 0 if "Healthy" in name else 1 for name in class_names}
    binary_remapped_dataset = RemappedDataset(full_dataset, binary_mapping)

    # Use stratified split for a more robust test set
    train_val_samples, test_samples, train_val_targets, test_targets = train_test_split(
        binary_remapped_dataset.samples, 
        binary_remapped_dataset.targets, 
        test_size=0.2, 
        stratify=binary_remapped_dataset.targets, 
        random_state=SEED
    )

    train_samples, val_samples, train_targets, val_targets = train_test_split(
        train_val_samples, 
        train_val_targets, 
        test_size=0.25,  # 0.25 of train_val is 0.2 of total
        stratify=train_val_targets, 
        random_state=SEED
    )

    train_dataset = RemappedDataset(full_dataset, binary_mapping, transform=binary_transforms['train'])
    val_dataset = RemappedDataset(full_dataset, binary_mapping, transform=binary_transforms['val_test'])
    test_dataset = RemappedDataset(full_dataset, binary_mapping, transform=binary_transforms['val_test'])

    train_dataset.samples = train_samples
    train_dataset.targets = train_targets
    val_dataset.samples = val_samples
    val_dataset.targets = val_targets
    test_dataset.samples = test_samples
    test_dataset.targets = test_targets
    
    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in train_dataset.targets]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Binary dataset preprocessing complete.")
    return train_loader, val_loader, test_loader, ["Healthy", "Diseased"]

def get_multiclass_dataset(dataset_dir=DATASET_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Loads dataset for multiclass classification (diseased leaves only)
    """
    print(f"Loading multiclass dataset from: {dataset_dir}")
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val_test': transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    full_dataset = datasets.ImageFolder(root=dataset_dir)
    original_classes = full_dataset.classes
    diseased_class_names = sorted([name for name in original_classes if "Healthy" not in name])
    new_class_to_idx = {name: i for i, name in enumerate(diseased_class_names)}

    # Filter dataset to only include diseased images
    diseased_samples = []
    diseased_targets = []
    for path, label in full_dataset.samples:
        class_name = full_dataset.classes[label]
        if "Healthy" not in class_name:
            diseased_samples.append((path, new_class_to_idx[class_name]))
            diseased_targets.append(new_class_to_idx[class_name])
    
    # Manually split the filtered dataset to ensure correct labels
    np.random.seed(SEED)
    np.random.shuffle(diseased_samples)
    
    train_size = int(0.6 * len(diseased_samples))
    val_size = int(0.2 * len(diseased_samples))
    test_size = len(diseased_samples) - train_size - val_size

    train_samples = diseased_samples[:train_size]
    val_samples = diseased_samples[train_size:train_size + val_size]
    test_samples = diseased_samples[train_size + val_size:]
    
    train_dataset = FilteredDataset(train_samples, transform=data_transforms['train'])
    val_dataset = FilteredDataset(val_samples, transform=data_transforms['val_test'])
    test_dataset = FilteredDataset(test_samples, transform=data_transforms['val_test'])

    train_labels = train_dataset.targets
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print("Multiclass dataset preprocessing complete.")
    return train_loader, val_loader, test_loader, diseased_class_names