import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Set path to import models and data_loader
sys.path.insert(0, '/home/palodo2/acl_classifier')
from src.data_loader import OptimizedMRNetDataset, get_val_test_transform
from src.models import CNNMultiSliceClassifier

def calculate_metrics(labels, probs, threshold=0.5):
    predictions = (probs >= threshold).astype(int)
    tn = ((predictions == 0) & (labels == 0)).sum()
    tp = ((predictions == 1) & (labels == 1)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {'precision': precision, 'recall': recall}

def find_optimal_threshold_recall(labels, probs, min_precision=0.75):
    thresholds = np.linspace(0, 1, 1000)
    best_threshold = 0.5
    best_recall = -1
    
    for thresh in thresholds:
        metrics = calculate_metrics(labels, probs, threshold=thresh)
        if metrics['precision'] >= min_precision:
            if metrics['recall'] > best_recall:
                best_recall = metrics['recall']
                best_threshold = thresh
    return best_threshold

def get_predictions(model, loader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
    return np.array(all_probs)

def main():
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    BASE_DIR = Path('/home/palodo2/acl_classifier')
    DATA_DIR = BASE_DIR / 'data'
    CHECKPOINTS_DIR = BASE_DIR / 'checkpoints'
    
    VAL_CSV = DATA_DIR / 'val-acl.csv'
    CNN_MULTISEED_DIR = CHECKPOINTS_DIR / 'cnn_multiseed_resnet50'
    
    PLANES = ['sagittal', 'coronal', 'axial']
    CNN_MODELS = {
        'sagittal': CNN_MULTISEED_DIR / 'best_sagittal_multiseed_final.pth',
        'coronal': CNN_MULTISEED_DIR / 'best_coronal_multiseed_final.pth',
        'axial': CNN_MULTISEED_DIR / 'best_axial_multiseed_final.pth'
    }
    
    test_transform = get_val_test_transform()
    val_predictions = {}
    val_labels = None
    
    # CNN config per plane
    CNN_PLANE_CONFIGS = {
        'sagittal': {'pooling_mode': 'attention', 'dropout_input': 0.42, 'dropout_dense': 0.32},
        'coronal': {'pooling_mode': 'max', 'dropout_input': 0.47, 'dropout_dense': 0.37},
        'axial': {'pooling_mode': 'attention', 'dropout_input': 0.42, 'dropout_dense': 0.32}
    }
    
    for plane in PLANES:
        print(f"Processing {plane}...")
        val_dataset = OptimizedMRNetDataset(
            csv_path=str(VAL_CSV),
            data_root=str(DATA_DIR / 'val'),
            plane=plane,
            indices_cache_path=str(DATA_DIR / 'slice_indices_final' / f'val_{plane}_indices.json'),
            transform=test_transform
        )
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        if val_labels is None:
            val_labels = val_dataset.df['label'].values
            
        cfg = CNN_PLANE_CONFIGS[plane]
        model = CNNMultiSliceClassifier(
            num_classes=1,
            pretrained=False,
            pooling_mode=cfg['pooling_mode'],
            dropout_input=cfg['dropout_input'],
            dropout_dense=cfg['dropout_dense']
        )
        
        checkpoint = torch.load(str(CNN_MODELS[plane]), map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
            
        model = model.to(device)
        probs = get_predictions(model, val_loader, device)
        val_predictions[plane] = probs
        
        print(f"  {plane} AUC: {roc_auc_score(val_labels, probs):.4f}")
        
    ensemble_probs = (val_predictions['sagittal'] + val_predictions['coronal'] + val_predictions['axial']) / 3.0
    print(f"Ensemble AUC: {roc_auc_score(val_labels, ensemble_probs):.4f}")
    
    best_thr = find_optimal_threshold_recall(val_labels, ensemble_probs, min_precision=0.75)
    print(f"CNN Ensemble Optimal Threshold (min_precision=0.75): {best_thr:.4f}")

if __name__ == '__main__':
    main()
