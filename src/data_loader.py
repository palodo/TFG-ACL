"""
Data loader module for ACL Classifier
Implements OptimizedMRNetDataset and transformations.
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class OptimizedMRNetDataset(Dataset):
    """
    Optimized dataset using pre-calculated slice indices.

    Advantages:
    - No need for CNN Selector at runtime
    - Instant slice loading
    - Fully reproducible
    - ~100x faster than on-the-fly index calculation
    - Supports plane-specific augmentation
    """
    def __init__(self, csv_path, data_root, plane, indices_cache_path, augmentation_mode=None, transform=None):
        """
        Args:
            csv_path: Path to CSV with labels
            data_root: Root directory with data (train/val/test)
            plane: 'sagittal', 'coronal', or 'axial'
            indices_cache_path: Path to JSON with pre-calculated indices
            augmentation_mode: 'conservative', 'moderate', 'aggressive' or None
            transform: Transforms to apply (overridable by augmentation_mode)
        """
        self.df = pd.read_csv(csv_path, header=None, names=['case', 'label'])
        self.data_root = Path(data_root)
        self.plane = plane

        # Load pre-calculated indices
        with open(indices_cache_path, 'r') as f:
            self.indices_cache = json.load(f)

        # Determine which transform to use
        if augmentation_mode:
            self.transform = get_train_transform(augmentation_mode)
        elif transform:
            self.transform = transform
        else:
            self.transform = get_val_test_transform()

        print(f" Dataset cargado: {len(self.df)} casos")
        print(f"  Plane: {plane}")
        print(f"  Augmentation: {augmentation_mode if augmentation_mode else 'None (usando transform directo)'}")
        print(f"  Índices cacheados: {indices_cache_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Return a case with selected slices and label.
        """
        row = self.df.iloc[idx]
        case_id = row['case']
        label = row['label']

        # Load full volume
        volume_path = self.data_root / self.plane / f"{case_id:04d}.npy"
        volume = np.load(volume_path).astype(np.float32)

        # Get pre-calculated indices for this case
        selected_indices = self.indices_cache[str(case_id)]

        # Extract only selected slices
        selected_slices = volume[selected_indices]  # (K, H, W)

        # Min-max normalization
        vol_min = selected_slices.min()
        vol_max = selected_slices.max()
        if vol_max > vol_min:
            selected_slices = (selected_slices - vol_min) / (vol_max - vol_min)

        # Convert to tensor and replicate to 3 channels
        selected_slices = torch.from_numpy(selected_slices).float()
        selected_slices = selected_slices.unsqueeze(1)  # (K, 1, H, W)
        selected_slices = selected_slices.repeat(1, 3, 1, 1)  # (K, 3, H, W)

        # Apply transformations if available
        if self.transform:
            transformed_slices = []
            for i in range(selected_slices.shape[0]):
                slice_transformed = self.transform(selected_slices[i])
                transformed_slices.append(slice_transformed)
            selected_slices = torch.stack(transformed_slices, dim=0)  # (K, 3, 224, 224)

        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return selected_slices, label_tensor, case_id


# ==========================================
# TRANSFORMATION FUNCTIONS
# ==========================================

def get_train_transform(augmentation_mode='conservative'):
    """
    Return training transformations based on augmentation mode.

    Args:
        augmentation_mode: 'conservative', 'moderate', or 'aggressive'

    Returns:
        transforms.Compose with appropriate transformations
    """
    if augmentation_mode == 'aggressive':
        # Aggressive augmentation - provides high variability
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.6),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_mode == 'moderate':
        # Moderate augmentation - balance between regularization and data variety
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Conservative augmentation - minimal perturbations, preserves anatomy
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=7),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.98, 1.02)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_val_test_transform():
    """
    Return validation/test transformations (no augmentation).
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
