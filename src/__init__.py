"""
ACL Classifier - Deep Learning for ACL Lesion Detection in MRI

Core modules for model architectures, data loading, and training utilities.
"""

__version__ = "1.0.0"
__author__ = "Palodo2 - Universidad de Valencia"

# Import key components for easy access
try:
    from .models import (
        ConvNeXtMultiSliceClassifier,
        SwinMultiSliceClassifier,
        CNNMultiSliceClassifier,
        AttentionPooling,
        SmoothedBCELoss
    )
    from .data_loader import OptimizedMRNetDataset, get_train_transform, get_val_test_transform
    from .training_utils import train_model
    from .gpu_config import device, VIT_GPU_IDS, CNN_GPU_ID
except ImportError as e:
    print(f"Warning: Could not import all modules. {e}")

__all__ = [
    'ConvNeXtMultiSliceClassifier',
    'SwinMultiSliceClassifier', 
    'CNNMultiSliceClassifier',
    'AttentionPooling',
    'SmoothedBCELoss',
    'OptimizedMRNetDataset',
    'get_train_transform',
    'get_val_test_transform',
    'train_model',
    'device',
    'VIT_GPU_IDS',
    'CNN_GPU_ID'
]
