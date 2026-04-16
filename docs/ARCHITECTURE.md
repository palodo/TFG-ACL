# Architecture Guide

## Overview

The ACL Classifier uses a multi-view ensemble approach combining three anatomical planes (sagittal, coronal, axial) with Vision Transformer models.

```
MRI Volume (3D)
    ↓
[Sagittal]  [Coronal]  [Axial]    (Plane-wise Slice Extraction)
    ↓          ↓           ↓
  ViT-T      ViT-T       ViT-T     (Independent Classification)
    ↓          ↓           ↓
  Prob       Prob        Prob      (Per-plane Probabilities)
    ↓          ↓           ↓
  [Ensemble Voting / Averaging]
    ↓
  Final Prediction (ACL Class)
```

## Model Components

### 1. Vision Transformer (ViT) Backbone

**Location**: `src/models.py`

```python
class ViTTinyMultiSliceClassifier(nn.Module):
    """
    Multi-slice ViT classifier for ACL detection
    """
```

**Architecture Details**:
- **Base Model**: `WinKawaks/vit-tiny-patch16-224` (HuggingFace)
- **Features**: 192-dimensional embeddings
- **Parameters**: 5.7M (ultra-compact)
- **Input**: 224×224 RGB images
- **Attention Layers**: 12 transformer blocks

### 2. Multi-Slice Processing

**Input Processing**:
- Extract 5-10 consecutive slices per plane
- Normalize to [0, 1] range
- Optional augmentation

**Aggregation Methods**:
1. **Attention Pooling** (Default)
   - Learned weight for each slice
   - Weight softmax normalization
   - Final: weighted average of features

2. **Mean Pooling**
   - Simple average across slices
   - Baseline method

3. **Max Pooling**
   - Takes maximum activation across slices

### 3. Classification Head

```python
# After feature aggregation:
x_pooled = attention_pool(slice_features)  # Shape: (batch, 192)
logits = classifier_head(x_pooled)         # Shape: (batch, 2)
probs = softmax(logits)                    # Shape: (batch, 2)
```

## Data Pipeline

### Slice Indexing

**Location**: `src/preprocessing.py`

For each MRI volume and plane:
1. Count total slices in volume
2. Select strategy-specific indices:
   - **center**: Middle N slices
   - **cnn_based**: CNN recommends important slices
   - **random**: Random slice selection (training only)

**Cached Format**: `data/slice_indices_final/{plane}_{case_id}.npy`

### Dataset Class

**Location**: `src/data_loader.py`

```python
class SlicedMRIDataset(Dataset):
    """
    Loads DICOM data and returns processed slices
    
    Returns:
        - images: Tensor of shape (num_slices, 3, 224, 224)
        - label: Binary classification {0, 1}
        - case_id: Unique identifier
    """
```

## Training Pipeline

### Configuration

**Location**: Notebooks in `models/*/`

```python
PLANE_SPECIFIC_CONFIG = {
    'sagittal': {
        'learning_rate': 1.5e-4,
        'batch_size': 16,
        'num_epochs': 100,
        'warmup_epochs': 5,
        'dropout': 0.3,
        'augmentation': 'conservative'
    },
    'coronal': {
        'learning_rate': 1.5e-4,
        'batch_size': 16,
        'num_epochs': 100,
        'warmup_epochs': 5,
        'dropout': 0.35,
        'augmentation': 'moderate'
    },
    'axial': {
        'learning_rate': 1.5e-4,
        'batch_size': 16,
        'num_epochs': 100,
        'warmup_epochs': 5,
        'dropout': 0.35,
        'augmentation': 'aggressive'
    }
}
```

### Training Loop

**Location**: `src/training_utils.py`

```python
def train_model(model, train_loader, val_loader, config):
    """
    Complete training pipeline with:
    - Warmup learning rate scheduling
    - Early stopping based on validation AUC
    - Gradient accumulation
    - Loss scaling for stability
    """
```

**Key Features**:
1. **Warmup**: Linear LR increase for first N epochs
2. **Scheduling**: CosineAnnealingLR after warmup
3. **Early Stopping**: Patient monitoring of validation AUC
4. **Gradient Clipping**: Norm = 1.0
5. **Mixed Precision**: FP16 training support (optional)

### Loss Functions

- **Training**: Cross-Entropy Loss (unweighted)
- **Validation**: Weighted Cross-Entropy (class weights)
- **Optimization**: AdamW with weight decay

## GPU Configuration

### Multi-GPU Setup

**Location**: `src/gpu_config.py`

```python
VIT_GPU_IDS = [0, 1, 2, 3]  # List of available GPUs
CNN_GPU_ID = 0               # GPU for preprocessing

# Usage:
model = nn.DataParallel(model, device_ids=VIT_GPU_IDS)
```

### Memory Management

- **Batch Size**: 16 per GPU (64 total on 4 GPUs)
- **Input Size**: 224×224 pixels
- **Max Slices**: 10 per plane
- **Expected Memory**: ~8GB per GPU

## Augmentation Strategies

### Conservative (Sagittal)
- SmallRandomCrop: 80% of image
- Rotation: ±10°
- Horizontal flip: 50%

### Moderate (Coronal)
- RandomCrop: 70% of image
- Rotation: ±15°
- Horizontal flip: 50%
- ColorJitter: ±0.2 brightness

### Aggressive (Axial)
- RandomCrop: 60% of image
- Rotation: ±20°
- Horizontal flip: 50%
- ColorJitter: ±0.3 brightness/contrast
- GaussianBlur: kernel 3-5

## Ensemble Strategy

### Voting Mechanism

```python
# Collect predictions from all 3 planes
pred_sagittal = model_sag.predict(sag_slices)     # (batch, 2)
pred_coronal = model_cor.predict(cor_slices)      # (batch, 2)
pred_axial = model_ax.predict(ax_slices)          # (batch, 2)

# Ensemble methods:
# 1. Average (default)
ensemble_prob = (pred_sagittal + pred_coronal + pred_axial) / 3
ensemble_pred = argmax(ensemble_prob)

# 2. Weighted average
weights = [0.4, 0.3, 0.3]  # Plane-specific weights
ensemble_prob = sum(p*w for p,w in zip(predictions, weights))
```

### Threshold Calibration

Options for final classification:
1. **Max F1**: Optimal balance
   - Equation: Threshold = argmax(2*TP/(2*TP+FP+FN))

2. **Max Recall**: Minimize False Negatives
   - Useful when missing injuries is critical

3. **Youden Index**: Sensitivity + Specificity
   - Equation: Threshold = argmax(TPR - FPR)

## Performance Metrics

### Evaluation Metrics

- **AUC-ROC**: Area Under ROC Curve
- **F1 Score**: Harmonic mean of precision/recall
- **Sensitivity**: True Positive Rate (recall)
- **Specificity**: True Negative Rate
- **Precision**: TP/(TP+FP)

### Cross-Validation

- **Strategy**: 5-fold stratified
- **Split**: Maintains class distribution
- **Reported**: Mean ± Std across folds

## Model Checkpoint

### Saving Strategy

```python
# Save best model based on validation AUC
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'best_auc': best_auc,
    'config': config
}
torch.save(checkpoint, f'checkpoints/{model_name}/best.pt')
```

### Loading Strategy

```python
checkpoint = torch.load('checkpoints/best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Inference Pipeline

### Single Volume Prediction

```python
1. Load MRI volume (DICOM)
2. Extract slices per plane using indices
3. Preprocess (normalize, resize to 224×224)
4. Forward pass through each plane model
5. Aggregate with ensemble voting
6. Apply calibrated threshold
7. Return: ACL Injury (YES/NO) + Confidence
```

### Batch Inference

```python
# Process multiple volumes efficiently
for case_id in case_ids:
    # Load volume
    sag_slices = load_slices(case_id, 'sagittal')
    cor_slices = load_slices(case_id, 'coronal')
    ax_slices = load_slices(case_id, 'axial')
    
    # GPU inference
    with torch.no_grad():
        logits_sag = model_sag(sag_slices)
        logits_cor = model_cor(cor_slices)
        logits_ax = model_ax(ax_slices)
    
    # Combine predictions
    final_pred = ensemble_vote([logits_sag, logits_cor, logits_ax])
```

---

**For detailed implementation, see source code in `src/` directory.**
