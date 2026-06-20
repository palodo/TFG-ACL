"""
Training utilities for ACL Classifier
Includes training loops, evaluation, and training orchestration.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score

from models import SmoothedBCELoss, WarmupCosineScheduler


def compute_pos_weight(train_loader):
    """
    Compute pos_weight for BCEWithLogitsLoss based on class imbalance.

    pos_weight = num_negatives / num_positives

    Args:
        train_loader: Training DataLoader

    Returns:
        pos_weight: tensor with computed weight
    """
    num_positives = 0
    num_negatives = 0

    for _, targets, _ in train_loader:
        targets = targets.squeeze()
        num_positives += (targets == 1).sum().item()
        num_negatives += (targets == 0).sum().item()

    if num_positives == 0:
        return torch.tensor(1.0)

    pos_weight = torch.tensor(num_negatives / num_positives, dtype=torch.float32)
    return pos_weight


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scheduler=None,
                use_grad_clip=False, grad_clip_norm=1.0):
    """
    Train one epoch.

    Args:
        model: Neural network model
        train_loader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: torch device
        epoch: Current epoch number
        scheduler: LR scheduler (optional)
        use_grad_clip: Whether to use gradient clipping
        grad_clip_norm: Gradient clipping norm

    Returns:
        avg_loss: Average loss for the epoch
        train_auc: AUC score for the epoch
    """
    model.train()
    total_loss = 0.0
    all_probs = []
    all_targets = []

    for batch_idx, (slices, targets, _) in enumerate(train_loader):
        slices = slices.to(device)
        targets = targets.to(device).unsqueeze(-1)

        # Forward pass
        outputs = model(slices)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

        # Compute loss
        loss = criterion(logits, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping if enabled
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        # Accumulate for AUC calculation
        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.flatten())
        all_targets.extend(targets.detach().cpu().numpy().flatten())

    avg_loss = total_loss / len(train_loader)
    train_auc = roc_auc_score(all_targets, all_probs)

    return avg_loss, train_auc


def evaluate(model, val_loader, criterion_objective, criterion_plain, device, split_name='val'):
    """
    Evaluate model on validation/test set.
    Returns multiple metrics.

    Args:
        model: Neural network model
        val_loader: Validation DataLoader
        criterion_objective: Loss with label smoothing
        criterion_plain: Plain BCE loss
        device: torch device
        split_name: Name of split (for logging)

    Returns:
        dict with metrics: loss_objective, loss_plain_bce, auc, f1, brier, ece
    """
    model.eval()

    total_loss_objective = 0.0
    total_loss_plain = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for slices, targets, _ in val_loader:
            slices = slices.to(device)
            targets = targets.to(device).unsqueeze(-1)

            # Forward pass
            outputs = model(slices)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            # Compute losses
            loss_obj = criterion_objective(logits, targets)
            loss_plain = criterion_plain(logits, targets)

            total_loss_objective += loss_obj.item()
            total_loss_plain += loss_plain.item()

            # Collect probabilities
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs.flatten())
            all_targets.extend(targets.detach().cpu().numpy().flatten())

    # Compute metrics
    avg_loss_objective = total_loss_objective / len(val_loader)
    avg_loss_plain = total_loss_plain / len(val_loader)
    auc = roc_auc_score(all_targets, all_probs)
    f1 = f1_score(all_targets, np.round(all_probs))

    # Brier Score (mean squared error)
    brier = np.mean((np.array(all_probs) - np.array(all_targets)) ** 2)

    # ECE (Expected Calibration Error) with 10 bins
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (np.array(all_probs) >= bin_boundaries[i]) & (np.array(all_probs) < bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_acc = np.mean(np.array(all_targets)[mask])
            bin_conf = np.mean(np.array(all_probs)[mask])
            ece += np.abs(bin_acc - bin_conf) * mask.sum() / len(all_targets)

    return {
        'loss_objective': avg_loss_objective,
        'loss_plain_bce': avg_loss_plain,
        'auc': auc,
        'f1': f1,
        'brier': brier,
        'ece': ece
    }


def train_model(model, train_loader, val_loader, num_epochs, device, save_dir, plane_name,
                learning_rate=1e-4, weight_decay=1e-4, label_smoothing=0.05,
                use_scheduler=False, use_grad_clip=False, grad_clip_norm=1.0, early_stopping_patience=15,
                dropout_input=0.3, dropout_dense=0.2, augmentation_mode='conservative',
                data_path=None, plane_config=None, use_wandb=False, wandb_run=None, warmup_epochs=3,
                use_lr_plateau_scheduler=False, lr_reduction_factor=0.5, lr_plateau_patience=5,
                freeze_backbone_epochs=0, pos_weight_factor=1.0):
    """
    Main training function for a single plane.
    Automatically selects hyperparameters based on plane configuration.
    Integrates Weights & Biases for experiment tracking.
    Supports ReduceLROnPlateau for learning rate annealing when AUC plateaus.
    Supports freeze_backbone_epochs for 2-stage transfer learning.

    Args:
        model: Neural network model (will be wrapped with DataParallel)
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Number of epochs to train
        device: torch device
        save_dir: Directory to save checkpoints
        plane_name: Name of plane ('sagittal', 'coronal', 'axial')
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        label_smoothing: Label smoothing parameter
        use_scheduler: Whether to use learning rate scheduler (WarmupCosineScheduler)
        use_grad_clip: Whether to use gradient clipping
        grad_clip_norm: Gradient clipping norm (default 1.0)
        early_stopping_patience: Patience for early stopping (epochs without AUC improvement)
        dropout_input: Dropout rate for input
        dropout_dense: Dropout rate for dense layers
        augmentation_mode: Augmentation strategy
        data_path: Path to data (for computing pos_weight if needed)
        plane_config: Configuration for the plane
        use_wandb: Whether to use Weights & Biases
        wandb_run: Existing wandb run object (optional)
        warmup_epochs: Number of warmup epochs for scheduler
        use_lr_plateau_scheduler: Whether to reduce LR when AUC plateaus (NEW)
        lr_reduction_factor: Factor to multiply LR by when plateau detected (default 0.5)
        lr_plateau_patience: Epochs without AUC improvement before reducing LR (default 5)
        freeze_backbone_epochs: Number of epochs to freeze backbone (only train classifier head) (NEW)

    Returns:
        history: dict with training history
        best_auc: Best validation AUC achieved
        best_f1: Best validation F1 achieved
    """

    MIN_LR = 1e-6

    print(f"\n{'='*80}")
    print(f"ENTRENANDO MODELO: {plane_name.upper()}")
    print(f"LR={learning_rate}, WD={weight_decay}, Scheduler={use_scheduler}, GradClip={use_grad_clip}")
    print(f"Dropout: Input={dropout_input}, Dense={dropout_dense}")
    print(f"Augmentation: {augmentation_mode}")
    print(f"{'='*80}\n")

    # Compute pos_weight for class imbalance (BEFORE wandb setup)
    pos_weight = compute_pos_weight(train_loader)
    pos_weight = pos_weight * pos_weight_factor  # Apply reduction factor
    print(f" Class imbalance (pos_weight): {pos_weight.item():.4f}")
    print(f"   (Ratio negatives:positives × {pos_weight_factor})\n")

    # Initialize wandb if enabled
    if use_wandb and wandb_run is None:
        try:
            import wandb
            wandb_config = {
                'plane': plane_name,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'label_smoothing': label_smoothing,
                'dropout_input': dropout_input,
                'dropout_dense': dropout_dense,
                'use_scheduler': use_scheduler,
                'augmentation_mode': augmentation_mode,
                'pos_weight': pos_weight.item(),
            }
            wandb_run = wandb.init(
                project="ACL-TFG",
                name=f"acl-{plane_name}-{augmentation_mode}",
                config=wandb_config
            )
            print(f" Weights & Biases inicializado para {plane_name}\n")
        except Exception as e:
            print(f"  Error al inicializar wandb: {e}\n")
            wandb_run = None

    # Setup loss functions with pos_weight
    criterion_train = SmoothedBCELoss(smoothing=label_smoothing, pos_weight=pos_weight)
    criterion_val_objective = SmoothedBCELoss(smoothing=label_smoothing, pos_weight=pos_weight)
    criterion_val_plain = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')

    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Setup scheduler
    scheduler = None
    if use_scheduler:
        scheduler = WarmupCosineScheduler(
            optimizer,
            total_epochs=num_epochs,
            warmup_epochs=warmup_epochs,
            min_lr=MIN_LR
        )

    # ===== FREEZE_BACKBONE SUPPORT (NEW) =====
    def freeze_backbone():
        """Freeze all parameters except the classifier head"""
        model_to_freeze = model.module if isinstance(model, nn.DataParallel) else model

        # Freeze backbone (all layers except the final classifier)
        for name, param in model_to_freeze.named_parameters():
            # Only freeze if it's NOT part of the classifier (head)
            if 'classifier' not in name and 'head' not in name:
                param.requires_grad = False

        print(f"   Backbone frozen - training only classifier head")

    def unfreeze_backbone():
        """Unfreeze all parameters for fine-tuning"""
        model_to_freeze = model.module if isinstance(model, nn.DataParallel) else model
        for param in model_to_freeze.parameters():
            param.requires_grad = True
        print(f"   Backbone unfrozen - full fine-tuning enabled")

    # Variables for early stopping and tracking
    best_auc = 0.0
    best_f1 = 0.0
    patience_counter = 0
    lr_plateau_counter = 0  # NEW: Counter for ReduceLROnPlateau
    best_auc_for_plateau = 0.0  # NEW: Track best AUC for plateau detection

    history = {
        'train_loss': [],
        'train_auc': [],
        'val_loss_objective': [],
        'val_loss_plain_bce': [],
        'val_auc': [],
        'val_f1': [],
        'val_brier': [],
        'val_ece': [],
        'learning_rates': []
    }

    # Training loop
    for epoch in range(1, num_epochs + 1):
        # ===== HANDLE BACKBONE FREEZING (NEW) =====
        if freeze_backbone_epochs > 0:
            if epoch == 1:
                freeze_backbone()
            elif epoch == freeze_backbone_epochs + 1:
                unfreeze_backbone()

        # Step scheduler BEFORE training (for warmup to apply from epoch 1)
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)

        print(f"Epoch {epoch}/{num_epochs} (LR={current_lr:.2e}):", end=' ')

        # Training
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion_train, optimizer, device, epoch,
            scheduler=scheduler, use_grad_clip=use_grad_clip, grad_clip_norm=grad_clip_norm
        )

        # Validation
        val_metrics = evaluate(model, val_loader, criterion_val_objective,
                              criterion_val_plain, device, split_name=plane_name)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss_objective'].append(val_metrics['loss_objective'])
        history['val_loss_plain_bce'].append(val_metrics['loss_plain_bce'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_brier'].append(val_metrics['brier'])
        history['val_ece'].append(val_metrics['ece'])

        # Print results
        print(f"Train: Loss={train_loss:.4f}, AUC={train_auc:.4f} | " +
              f"Val: Loss={val_metrics['loss_objective']:.4f}, AUC={val_metrics['auc']:.4f}, " +
              f"F1={val_metrics['f1']:.4f}", end='')

        # Log to wandb
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_auc': train_auc,
                'val_loss': val_metrics['loss_objective'],
                'val_auc': val_metrics['auc'],
                'val_f1': val_metrics['f1'],
                'learning_rate': current_lr
            })

        # ===== ReduceLROnPlateau: Reduce LR when AUC plateaus (NEW) =====
        if use_lr_plateau_scheduler:
            if val_metrics['auc'] > best_auc_for_plateau:
                best_auc_for_plateau = val_metrics['auc']
                lr_plateau_counter = 0
            else:
                lr_plateau_counter += 1

                if lr_plateau_counter >= lr_plateau_patience:
                    old_lr = optimizer.param_groups[0]['lr']
                    new_lr = old_lr * lr_reduction_factor
                    new_lr = max(new_lr, 1e-7)  # Never go below 1e-7

                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr

                    print(f"  LR Reduced: {old_lr:.2e} → {new_lr:.2e}", end='')
                    lr_plateau_counter = 0
                    best_auc_for_plateau = val_metrics['auc']

        # Early stopping - save best checkpoint
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_f1 = val_metrics['f1']
            patience_counter = 0

            # Get model state (handle DataParallel)
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_metrics['auc'],
                'val_f1': val_metrics['f1'],
                'val_loss': val_metrics['loss_objective'],
                'history': history
            }

            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()

            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"best_{plane_name}_auc.pth"
            torch.save(checkpoint, save_path)
            print(f"  Best AUC!", end='')
        else:
            patience_counter += 1

        print()

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n⏹  Early stopping: sin mejora en {early_stopping_patience} epochs.")
            break

    print(f"\n{'='*80}")
    print(f" ENTRENAMIENTO COMPLETADO: {plane_name.upper()}")
    print(f"  Mejor AUC: {best_auc:.4f}")
    print(f"  Mejor F1:  {best_f1:.4f}")
    print(f"  Augmentation: {augmentation_mode}")
    print(f"{'='*80}\n")

    # Finalize wandb
    if wandb_run is not None:
        wandb_run.summary['best_auc'] = best_auc
        wandb_run.summary['best_f1'] = best_f1
        wandb_run.finish()

    return history, best_auc, best_f1
