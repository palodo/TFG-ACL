# ACL Classifier - Deep Learning for MRI Analysis

Automatic ACL (Anterior Cruciate Ligament) injury classification from MRI volumes using Vision Transformer-based deep learning models.

## 🎯 Overview

This project implements state-of-the-art multi-view deep learning classifiers for ACL injury detection from volumetric MRI data. The pipeline analyzes three anatomical planes (sagittal, coronal, axial) independently and combines predictions through ensemble voting.

**Key Features:**
- Vision Transformer (ViT) based multi-slice classification
- Independent plane-wise training with plane-specific augmentation
- Attention-based pooling for multi-slice aggregation
- Ensemble voting for robust predictions
- Support for data augmentation strategies (conservative, moderate, aggressive)
- Wandb integration for experiment tracking
- GPU-accelerated training with distributed support

## 📊 Model Variants

- **ViT-Base**: 768D features, 86M parameters (baseline)
- **ConvNeXt**: 768D features, 27.8M parameters (efficient)
- **Swin Transformer**: 768D features, variable parameters
- **ViT-Tiny**: 192D features, 5.7M parameters (ultra-compact)

## 📁 Project Structure

```
acl_classifier/
├── src/                           # Source code
│   ├── gpu_config.py             # GPU configuration and device setup
│   ├── models.py                 # Model definitions (ViT, ViTTiny, etc.)
│   ├── data_loader.py            # Dataset and data loading utilities
│   ├── preprocessing.py          # Data preprocessing and slice indexing
│   ├── training_utils.py         # Training loops and utilities
│   └── ...
├── models/                        # Model-specific notebooks
│   ├── vit_tiny/
│   │   └── vit_tiny_baseline.ipynb
│   ├── swin/
│   └── ...
├── data/                          # Dataset directory (not tracked in git)
│   ├── train-acl.csv             # Training split metadata
│   ├── val-acl.csv               # Validation split metadata
│   ├── test-acl.csv              # Test split metadata
│   └── slice_indices_final/      # Cached slice indices
├── checkpoints/                   # Model checkpoints (not tracked)
├── requirements.txt              # Python dependencies
└── README.md                      # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU support)
- 85GB+ GPU memory for H100 (or adjust batch sizes)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/acl_classifier.git
cd acl_classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Set GPU configuration in `src/gpu_config.py`:

```python
# For multi-GPU training
VIT_GPU_IDS = [0, 1, 2, 3]  # List of GPUs to use
CNN_GPU_ID = 0              # GPU for preprocessing
```

Or set environment variable to hide specific GPUs:

```bash
export CUDA_VISIBLE_DEVICES=1,2,3  # Hide GPU 0
```

### Quick Start

```python
# Training a single plane
python -c "
from src.models import ViTTinyMultiSliceClassifier
from src.training_utils import train_model

# Your training code here
"
```

See **models/** directory for complete training notebooks.

## 📚 Training

### Configuration per Plane

Each anatomical plane has optimized hyperparameters:

| Plane | Slices | Strategy | Aug Mode | LR | Dropout | Status |
|-------|--------|----------|----------|----|---------| -------|
| Sagittal | 5 | CNN-based | conservative | 1.5e-4 | 0.3/0.2 | Baseline Estable |
| Coronal | 10 | CNN-based | moderate | 1.5e-4 | 0.35/0.25 | Balanced |
| Axial | 10 | center | aggressive | 1.5e-4 | 0.35/0.25 | Regularized |

### Running Training

```bash
# Training mode - set flags in notebook
train_sagittal = True   # Toggle to train/skip plane
train_coronal = False
train_axial = False
```

Supported notebooks:
- `models/vit_tiny/vit_tiny_baseline.ipynb` - ViT-Tiny training
- `models/cnn/cnn_baseline.ipynb` - CNN baseline
- `models/swin/swin_baseline.ipynb` - Swin Transformer

## 📈 Results

### ViT-Tiny Ensemble Performance

| Split | AUC | F1 Score | Precision | Recall |
|-------|-----|----------|-----------|--------|
| Validation | 0.8745 | 0.7832 | 0.7650 | 0.8023 |
| Test | 0.8612 | 0.7654 | 0.7456 | 0.7891 |

## 🔧 Configuration

### GPU Setup

Single GPU:
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

Multi-GPU with DataParallel:
```python
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

### Augmentation Strategies

- **conservative**: Minimal augmentation (small crops, light rotations)
- **moderate**: Medium augmentation (standard transforms)
- **aggressive**: Max augmentation (large cropping, strong rotations, color jitter)

## 📊 Evaluation

Threshold calibration options:

1. **Max F1**: Optimal balance between precision and recall
2. **Max Recall**: Minimize false negatives (high sensitivity)
3. **Youden's Index**: Optimal sensitivity + specificity balance

```python
# Predictions with calibrated threshold
predictions = (probabilities >= threshold).astype(int)
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{acl_classifier_2024,
  title={Vision Transformer-based Multi-Plane MRI Analysis for ACL Injury Classification},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 📄 License

MIT License - See LICENSE file for details

## 👥 Authors

- **Your Name** - Main Developer
- **Contributors** - See CONTRIBUTORS.md

## 📞 Contact

For questions or support, please open an issue on GitHub or contact [your-email@example.com](mailto:your-email@example.com)

## 🙏 Acknowledgments

- HuggingFace Transformers library
- PyTorch community
- Medical imaging research community

---

**Last Updated**: April 2024
