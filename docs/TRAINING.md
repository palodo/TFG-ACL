# Training Guide

Complete guide for training ACL Classifier models from scratch.

## Prerequisites

- **GPU**: H100 with 80GB memory recommended (adjust batch sizes for smaller GPUs)
- **CUDA**: 11.8+
- **Python**: 3.9-3.11
- **Dependencies**: Installed via `pip install -r requirements.txt`

## Quick Start

### 1. Prepare Data

```bash
# Ensure data structure is:
data/
├── train-acl.csv          # (case_id, label)
├── val-acl.csv
├── test-acl.csv
├── processed_dicom/       # Case folders with DICOM slices
└── slice_indices_final/   # Pre-extracted slice indices
```

### 2. Configure Training

Edit `models/vit_tiny/vit_tiny_baseline.ipynb` Cell 2 (Configuration):

```python
PLANE_CONFIG = {
    'sagittal': {'num_slices': 5, 'strategy': 'cnn_based'},
    'coronal': {'num_slices': 10, 'strategy': 'cnn_based'},
    'axial': {'num_slices': 10, 'strategy': 'center_consecutive'}
}

PLANE_SPECIFIC_CONFIG = {
    'sagittal': {
        'learning_rate': 1.5e-4,
        'batch_size': 16,
        'num_epochs': 100,
        'warmup_epochs': 5,
    },
    # ... configure per plane
}
```

### 3. Run Training

```bash
cd /path/to/acl_classifier
python -m jupyter notebook models/vit_tiny/vit_tiny_baseline.ipynb
```

Or execute from Python:

```python
import sys
sys.path.append('/path/to/acl_classifier')

from src.models import ViTTinyMultiSliceClassifier
from src.training_utils import train_model
from src.data_loader import SlicedMRIDataset

# Load data
train_dataset = SlicedMRIDataset('data/train-acl.csv', split='train')
val_dataset = SlicedMRIDataset('data/val-acl.csv', split='val')

# Create model
model = ViTTinyMultiSliceClassifier(
    num_slices=5,
    attention_pooling=True
)

# Train
history = train_model(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config={
        'learning_rate': 1.5e-4,
        'batch_size': 16,
        'num_epochs': 100,
        'warmup_epochs': 5,
    }
)
```

## Detailed Configuration

### Hyperparameters by Plane

#### Sagittal Plane

```python
{
    'num_slices': 5,
    'slice_strategy': 'cnn_based',
    'learning_rate': 1.5e-4,
    'batch_size': 16,
    'num_epochs': 100,
    'warmup_epochs': 5,
    'early_stopping_patience': 25,
    'weight_decay': 4.5e-4,
    'label_smoothing': 0.0,
    'dropout_backbone': 0.3,
    'dropout_head': 0.2,
    'use_scheduler': True,
    'use_grad_clip': True,
    'grad_clip_norm': 1.0,
    'augmentation': 'conservative',
    'freeze_backbone_epochs': 0,
}
```

**Rationale**:
- Conservative augmentation prevents overfitting on limited sagittal data
- Lower dropout for stable training
- No label smoothing for clear decision boundaries

#### Coronal Plane

```python
{
    'num_slices': 10,
    'slice_strategy': 'cnn_based',
    'learning_rate': 1.5e-4,
    'batch_size': 16,
    'num_epochs': 100,
    'warmup_epochs': 5,
    'early_stopping_patience': 20,
    'weight_decay': 4.5e-4,
    'label_smoothing': 0.0,
    'dropout_backbone': 0.35,
    'dropout_head': 0.25,
    'use_scheduler': True,
    'use_grad_clip': True,
    'augmentation': 'moderate',
    'freeze_backbone_epochs': 0,
}
```

**Rationale**:
- Moderate augmentation balances regularization
- More slices → slightly higher dropout
- Balanced hyperparameters

#### Axial Plane

```python
{
    'num_slices': 10,
    'slice_strategy': 'center_consecutive',
    'learning_rate': 1.5e-4,
    'batch_size': 16,
    'num_epochs': 100,
    'warmup_epochs': 5,
    'early_stopping_patience': 20,
    'weight_decay': 4.5e-4,
    'label_smoothing': 0.1,
    'dropout_backbone': 0.35,
    'dropout_head': 0.25,
    'use_scheduler': True,
    'use_grad_clip': True,
    'augmentation': 'aggressive',
    'freeze_backbone_epochs': 0,
}
```

**Rationale**:
- Aggressive augmentation for regularization
- Label smoothing (0.1) for soft targets
- Highest regularization on axial plane

### Learning Rate Scheduling

#### Warmup Phase

```
LR(epoch) = min_lr + (max_lr - min_lr) * epoch / warmup_epochs
```

- **Duration**: 5 epochs
- **Start LR**: 1e-5
- **Target LR**: 1.5e-4

#### Main Training Phase

```
LR(epoch) = lr_min + (lr_max - lr_min) * (1 + cos(π * epoch / T)) / 2
```

After warmup:
- **Algorithm**: Cosine Annealing
- **T_max**: num_epochs - warmup_epochs
- **lr_min**: 1e-5
- **lr_max**: 1.5e-4

### Data Augmentation

#### Conservative (Sagittal)

```python
transforms = [
    SmallRandomCrop(0.8),         # 80% crop
    RandomRotate(degrees=10),
    RandomHorizontalFlip(0.5),
]
```

#### Moderate (Coronal)

```python
transforms = [
    RandomCrop(0.7),              # 70% crop
    RandomRotate(degrees=15),
    RandomHorizontalFlip(0.5),
    ColorJitter(0.2, 0.2, 0.2),
]
```

#### Aggressive (Axial)

```python
transforms = [
    RandomCrop(0.6),              # 60% crop
    RandomRotate(degrees=20),
    RandomHorizontalFlip(0.5),
    ColorJitter(0.3, 0.3, 0.3),
    GaussianBlur(kernel_size=[3, 5]),
    RandomErasing(p=0.1),
]
```

## Training Dynamics

### Loss Function

```python
# During training:
loss_train = CrossEntropyLoss(reduction='mean')

# During validation (for monitoring):
class_weight = compute_class_weight('balanced', classes=[0, 1], y=labels)
loss_val = CrossEntropyLoss(weight=class_weight)
```

### Optimization Strategy

```python
optimizer = AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=4.5e-4,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

### Early Stopping

```python
# Monitor validation AUC
if val_auc > best_auc:
    best_auc = val_auc
    patience_counter = 0
    save_checkpoint(model, epoch)
else:
    patience_counter += 1
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

Patience values per plane:
- **Sagittal**: 25 epochs
- **Coronal**: 20 epochs
- **Axial**: 20 epochs

## GPU Setup

### Single GPU

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model = ViTTinyMultiSliceClassifier()
model.to('cuda:0')
```

### Multi-GPU (DataParallel)

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

model = ViTTinyMultiSliceClassifier()
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
model.to('cuda:0')
```

### Memory Optimization

**Batch Size per GPU**: 16
**Total Batch Size**: 64 (on 4 GPUs)

For smaller GPUs:
```python
# Adjust batch size
batch_size = 8 or 4  # Smaller GPUs
num_gpus = 1 or 2

total_batch = batch_size * num_gpus
```

**Gradient Accumulation** (if needed):

```python
accumulation_steps = 4
for batch_idx, (images, labels) in enumerate(train_loader):
    outputs = model(images)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Monitoring Training

### Metrics Tracked

- **Loss**: Training and validation cross-entropy
- **AUC-ROC**: Area under ROC curve
- **F1 Score**: Harmonic mean of precision/recall
- **Accuracy**: Percentage correct predictions

### Weights & Biases Integration

```python
import wandb

wandb.init(
    project="acl-classifier",
    name=f"vit-tiny-{plane}",
    config=config
)

wandb.log({
    'loss': loss,
    'auc': auc,
    'f1': f1,
    'learning_rate': scheduler.get_last_lr()[0]
})
```

### TensorBoard Integration (Alternative)

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(f'runs/{plane}')
writer.add_scalar('Loss/train', loss, epoch)
writer.add_scalar('AUC/val', auc, epoch)
```

## Troubleshooting

### Problem: Out of Memory (OOM)

**Solution**:
```python
# Reduce batch size
batch_size = 8  # Instead of 16

# Or use gradient accumulation
accumulation_steps = 2
```

### Problem: Validation AUC Not Improving

**Check**:
1. Learning rate too high/low → Adjust warmup
2. Augmentation too aggressive → Reduce intensity
3. Batch size too small → Increase if memory allows
4. Model underfitting → Increase num_epochs

**Fix**:
```python
# Gentler augmentation
'augmentation': 'conservative'

# Longer warmup
'warmup_epochs': 10

# Larger batch size
'batch_size': 32
```

### Problem: Training Loss Oscillating

**Solution**:
```python
# Enable gradient clipping
'use_grad_clip': True
'grad_clip_norm': 1.0

# Reduce learning rate
'learning_rate': 1.0e-4  # Was 1.5e-4
```

### Problem: Model Overfitting

**Solutions**:
```python
# Increase regularization
'weight_decay': 6.0e-4,
'dropout_backbone': 0.4,
'label_smoothing': 0.15,

# Stronger augmentation
'augmentation': 'aggressive'

# Early stopping
'early_stopping_patience': 15  # More aggressive
```

## Saving & Loading Models

### Save Checkpoint

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_auc': best_auc,
    'config': config
}

torch.save(checkpoint, 'checkpoints/vit_tiny_sagittal_best.pt')
```

### Load Checkpoint

```python
checkpoint = torch.load('checkpoints/vit_tiny_sagittal_best.pt')

model = ViTTinyMultiSliceClassifier(**config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Optionally: Resume training
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
```

## Hyperparameter Tuning

### Suggested Grid Search

```python
learning_rates = [1.0e-4, 1.5e-4, 2.0e-4]
weight_decays = [1.0e-4, 4.5e-4, 1.0e-3]
warmup_epochs = [0, 5, 10]

# Search for best combination per plane
```

### Expected Results

```
Model: ViT-Tiny (5.7M params)
Plane  | AUC   | F1    | Accuracy
-------|-------|-------|----------
Sag    | 0.87  | 0.78  | 82%
Cor    | 0.84  | 0.75  | 79%
Axial  | 0.81  | 0.72  | 76%
Ens    | 0.88  | 0.79  | 83%
```

---

**Next Steps**: See [RESULTS.md](RESULTS.md) for benchmark comparisons.
