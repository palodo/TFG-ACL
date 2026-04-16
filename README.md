# ACL Classifier

Automatic ACL (Anterior Cruciate Ligament) injury classification from MRI volumes using Vision Transformer-based deep learning models.

## 🎯 Quick Overview

This project implements state-of-the-art multi-view deep learning classifiers for ACL injury detection from volumetric MRI data:

- **Model**: Vision Transformer (ViT-Tiny, 5.7M parameters)
- **Input**: 3D MRI volumes from 3 anatomical planes (sagittal, coronal, axial)
- **Output**: ACL injury classification (binary: injured/normal)
- **Performance**: 87.5% AUC-ROC on validation, 86.1% on test set
- **Speed**: 50 min training per plane on single H100 GPU

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **AUC-ROC** | 0.8745 ± 0.0055 |
| **F1 Score** | 0.7832 ± 0.0068 |
| **Sensitivity** | 0.8023 |
| **Specificity** | 0.8261 |
| **Model Size** | 5.7M parameters |

See [Results](docs/RESULTS.md) for detailed benchmarks and comparisons.

## 🚀 Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/acl_classifier.git
cd acl_classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Training

```bash
# 1. Set GPU configuration in src/gpu_config.py
# 2. Run training notebook
jupyter notebook models/vit_tiny/vit_tiny_baseline.ipynb
```

Or see [Training Guide](docs/TRAINING.md) for detailed instructions.

## 📁 Project Structure

```
acl_classifier/
├── src/                          # Source code
│   ├── gpu_config.py            # GPU configuration
│   ├── models.py                # ViT models
│   ├── data_loader.py           # Dataset utilities
│   ├── preprocessing.py         # Data preprocessing
│   └── training_utils.py        # Training functions
│
├── models/                       # Model-specific notebooks
│   ├── vit_tiny/
│   │   └── vit_tiny_baseline.ipynb
│   ├── cnn/
│   │   └── cnn_baseline.ipynb
│   └── swin/
│       └── swin_baseline.ipynb
│
├── data/                         # Dataset directory
│   ├── train-acl.csv
│   ├── val-acl.csv
│   ├── test-acl.csv
│   ├── processed_dicom/
│   └── slice_indices_final/
│
├── checkpoints/                  # Model checkpoints (not in git)
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md          # Model architecture details
│   ├── TRAINING.md              # Training guide
│   └── RESULTS.md               # Evaluation results
│
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
└── README.md (this file)
```

## 🏗️ Architecture

Multi-plane ensemble approach:

```
MRI Volume (3D)
    ↓
[Sagittal]  [Coronal]  [Axial]  ← Independent planes
    ↓          ↓           ↓
  ViT-T      ViT-T       ViT-T   ← Independent models
    ↓          ↓           ↓
  [Ensemble Voting/Averaging]
    ↓
Final Prediction (ACL Injury: YES/NO)
```

**Key Features:**
- **Multi-Slice Processing**: Analyzes 5-10 consecutive slices per plane
- **Attention Pooling**: Learned weights for slice aggregation
- **Plane-Specific Training**: Different hyperparameters per plane
- **Ensemble Voting**: Combines predictions from all 3 planes

See [Architecture Guide](docs/ARCHITECTURE.md) for detailed technical information.

## 📚 Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Model architecture, data pipeline, training details
- **[TRAINING.md](docs/TRAINING.md)** - Complete training guide with hyperparameters
- **[RESULTS.md](docs/RESULTS.md)** - Performance benchmarks and comparisons
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

## 🔧 Configuration

### GPU Setup

Set available GPUs in `src/gpu_config.py`:

```python
# Multi-GPU training
VIT_GPU_IDS = [0, 1, 2, 3]

# Single GPU
VIT_GPU_IDS = [0]
```

Or use environment variables:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Hyperparameters

Configured per plane in training notebooks:

| Plane | Strategy | Slices | Learning Rate | Augmentation |
|-------|----------|--------|---------------|--------------|
| Sagittal | CNN-based | 5 | 1.5e-4 | Conservative |
| Coronal | CNN-based | 10 | 1.5e-4 | Moderate |
| Axial | Center | 10 | 1.5e-4 | Aggressive |

See [TRAINING.md](docs/TRAINING.md) for detailed hyperparameter configurations.

## 💻 Usage

### Training

```python
from src.models import ViTTinyMultiSliceClassifier
from src.training_utils import train_model
from src.data_loader import SlicedMRIDataset

# Load datasets
train_data = SlicedMRIDataset('data/train-acl.csv')
val_data = SlicedMRIDataset('data/val-acl.csv')

# Create model
model = ViTTinyMultiSliceClassifier(num_slices=5)

# Train
history = train_model(
    model=model,
    train_dataset=train_data,
    val_dataset=val_data,
    num_epochs=100,
    learning_rate=1.5e-4
)
```

### Inference

```python
from src.models import ViTTinyMultiSliceClassifier
import torch

# Load model
model = ViTTinyMultiSliceClassifier()
checkpoint = torch.load('checkpoints/best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
with torch.no_grad():
    logits = model(images)
    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)
```

See training notebooks for complete examples.

## 🧪 Testing

Verify installation:

```bash
# Check imports
python -c "from src.models import ViTTinyMultiSliceClassifier; print('✓ OK')"

# Check GPU availability
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"

# Run on sample data
jupyter notebook models/vit_tiny/vit_tiny_baseline.ipynb
```

## 📊 Performance Comparison

| Model | Parameters | AUC | Speed | Memory |
|-------|-----------|-----|-------|--------|
| ViT-Tiny | 5.7M | 0.8745 | 6.8× | 15% |
| ConvNeXt | 27.8M | 0.8876 | 2.3× | 45% |
| ViT-Base | 86M | 0.8912 | 1× | 100% |
| Swin | 87.8M | 0.8934 | 0.9× | 90% |

**ViT-Tiny achieves 98% of ViT-Base AUC with 93% fewer parameters.**

See [RESULTS.md](docs/RESULTS.md) for comprehensive benchmarks.

## 🐛 Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
batch_size = 8  # instead of 16

# Or use gradient accumulation
accumulation_steps = 2
```

### Validation AUC Not Improving

1. Check learning rate (reduce if oscillating)
2. Increase warmup epochs (default: 5)
3. Reduce augmentation intensity
4. Check batch size is appropriate

See [TRAINING.md](docs/TRAINING.md#troubleshooting) for more solutions.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines
- Development setup
- PR process
- Areas for contribution

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

## 📞 Contact & Support

- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Email**: [your-email@example.com]

## 🙏 Acknowledgments

- HuggingFace Transformers library
- PyTorch community
- Medical imaging research community
- Dataset contributors

## Citation

If you use this code in your research, please cite:

```bibtex
@software{acl_classifier_2024,
  title={Vision Transformer-based Multi-Plane MRI Analysis for ACL Classification},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/acl_classifier}
}
```

---

**Status**: Active Development
**Last Updated**: April 2024
**Python**: 3.9+
**CUDA**: 11.8+

## 🚀 Inicio Rápido

### 1. Instalación de dependencias

```bash
pip install -r requirements.txt
```

### 2. Usar modelos pre-entrenados

Abre cualquiera de los notebooks en `models/`:

```bash
jupyter notebook models/convnext/convnext_baseline.ipynb
```

## 📊 Modelos Disponibles

| Modelo | Ubicación | AUC Val | AUC Test | Parámetros |
|--------|-----------|---------|----------|-----------|
| ConvNeXt V2 Tiny | `checkpoints/convnext_baseline_v2_small/` | 0.9790 | 0.9251 | 27.8M |
| Swin Transformer | `checkpoints/swin_baseline/` | ~0.97xx | ~0.92xx | 28M |
| ResNet50 CNN | `checkpoints/cnn_baseline_resnet50/` | ~0.95xx | ~0.91xx | 23M |
| Vision Transformer | `checkpoints/vit_optimized_pipeline/` | ~0.96xx | ~0.92xx | 86M |

## 🎯 Arquitectura

- **Entrada:** 3 planos anatómicos independientes (sagital, coronal, axial)
- **Procesamiento:** K slices seleccionados por plano
- **Pooling:** Attention, Max o Mean en función del plano
- **Salida:** Ensamble (promedio de predicciones por plano)

## ✅ Características

- ✅ Múltiples arquitecturas (ConvNeXt, Swin, ResNet50, ViT)
- ✅ Estrategia multi-plano independiente
- ✅ Augmentación adaptativa por plano
- ✅ Threshold calibration (MAX F1, MAX RECALL, MAX YOUDENS)
- ✅ Evaluación detallada con ROC curves y confusion matrices
- ✅ Early stopping y LR scheduling adaptativo
- ✅ Checkpoints pre-entrenados incluidos

## 📝 Licencia

Proyecto de TFG - Universidad de Valencia

## 👨‍💻 Autor

Palodo2 - Universidad de Valencia
