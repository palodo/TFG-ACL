import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
import torch

# Set path to import models and data_loader
sys.path.insert(0, '/home/palodo2/acl_classifier')
from src.data_loader import OptimizedMRNetDataset, get_val_test_transform
from src.models import SwinMultiSliceClassifier, ViTMultiSliceClassifier, CNNMultiSliceClassifier

def get_predictions(model, loader, device):
    model.eval()
    all_probs = []
    all_case_ids = []
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            case_ids = batch[2]
            outputs = model(images)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_case_ids.extend(case_ids)
    return np.array(all_probs), all_case_ids

def main():
    # Detect GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    BASE_DIR = Path('/home/palodo2/acl_classifier')
    DATA_DIR = BASE_DIR / 'data'
    CHECKPOINTS_DIR = BASE_DIR / 'checkpoints'
    
    TEST_CSV = DATA_DIR / 'test-acl.csv'
    
    # Model checkpoints paths
    SWIN_MULTISEED_DIR = CHECKPOINTS_DIR / 'swin_small_multiseed'
    VIT_MULTISEED_DIR = CHECKPOINTS_DIR / 'vit_multiseed_base'
    CNN_MULTISEED_DIR = CHECKPOINTS_DIR / 'cnn_multiseed_resnet50'
    
    SWIN_MODELS = {
        'sagittal': SWIN_MULTISEED_DIR / 'best_sagittal_multiseed_final.pth',
        'coronal': SWIN_MULTISEED_DIR / 'best_coronal_multiseed_final.pth',
        'axial': SWIN_MULTISEED_DIR / 'best_axial_multiseed_final.pth'
    }
    
    VIT_MODELS = {
        'sagittal': VIT_MULTISEED_DIR / 'best_sagittal_multiseed_final.pth',
        'coronal': VIT_MULTISEED_DIR / 'best_coronal_multiseed_final.pth',
        'axial': VIT_MULTISEED_DIR / 'best_axial_multiseed_final.pth'
    }
    
    CNN_MODELS = {
        'sagittal': CNN_MULTISEED_DIR / 'best_sagittal_multiseed_final.pth',
        'coronal': CNN_MULTISEED_DIR / 'best_coronal_multiseed_final.pth',
        'axial': CNN_MULTISEED_DIR / 'best_axial_multiseed_final.pth'
    }
    
    # Thresholds
    THRESHOLDS = {
        'swin': 0.3734,
        'vit': 0.5135,
        'cnn': 0.5455
    }
    
    PLANES = ['sagittal', 'coronal', 'axial']
    test_transform = get_val_test_transform()
    
    # Load labels
    test_df = pd.read_csv(TEST_CSV, header=None, names=['case', 'label'])
    test_labels = test_df['label'].values
    case_ids_list = test_df['case'].apply(lambda x: f"{x:04d}").values
    num_cases = len(test_df)
    
    print(f"Total test cases: {num_cases}")
    
    # Dicts to store probabilities: model_name -> plane -> array of probs
    all_predictions = {
        'swin': {},
        'vit': {},
        'cnn': {}
    }
    
    # CNN config per plane
    cnn_configs = {
        'sagittal': {'pooling_mode': 'attention', 'dropout_input': 0.42, 'dropout_dense': 0.32},
        'coronal': {'pooling_mode': 'max', 'dropout_input': 0.47, 'dropout_dense': 0.37},
        'axial': {'pooling_mode': 'attention', 'dropout_input': 0.42, 'dropout_dense': 0.32}
    }
    
    # ViT config per plane
    vit_configs = {
        'sagittal': {'pooling_mode': 'attention', 'dropout_input': 0.3, 'dropout_dense': 0.2},
        'coronal': {'pooling_mode': 'max', 'dropout_input': 0.35, 'dropout_dense': 0.25},
        'axial': {'pooling_mode': 'attention', 'dropout_input': 0.35, 'dropout_dense': 0.25}
    }
    
    # Evaluate plane by plane
    for plane in PLANES:
        print(f"\n--- Processing plane: {plane} ---")
        test_dataset = OptimizedMRNetDataset(
            csv_path=str(TEST_CSV),
            data_root=str(DATA_DIR / 'test'),
            plane=plane,
            indices_cache_path=str(DATA_DIR / 'slice_indices_final' / f'test_{plane}_indices.json'),
            transform=test_transform
        )
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # 1. Swin Small
        print("  Evaluating Swin Small...")
        swin_model = SwinMultiSliceClassifier(pretrained=False).to(device)
        checkpoint = torch.load(str(SWIN_MODELS[plane]), map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        swin_model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
        swin_probs, _ = get_predictions(swin_model, test_loader, device)
        all_predictions['swin'][plane] = swin_probs
        del swin_model, checkpoint
        
        # 2. ViT Base
        print("  Evaluating ViT Base...")
        cfg_vit = vit_configs[plane]
        vit_model = ViTMultiSliceClassifier(
            pretrained=False,
            pooling_mode=cfg_vit['pooling_mode'],
            dropout_input=cfg_vit['dropout_input'],
            dropout_dense=cfg_vit['dropout_dense']
        ).to(device)
        checkpoint = torch.load(str(VIT_MODELS[plane]), map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        vit_model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
        vit_probs, _ = get_predictions(vit_model, test_loader, device)
        all_predictions['vit'][plane] = vit_probs
        del vit_model, checkpoint
        
        # 3. CNN ResNet50
        print("  Evaluating CNN ResNet50...")
        cfg_cnn = cnn_configs[plane]
        cnn_model = CNNMultiSliceClassifier(
            pretrained=False,
            pooling_mode=cfg_cnn['pooling_mode'],
            dropout_input=cfg_cnn['dropout_input'],
            dropout_dense=cfg_cnn['dropout_dense']
        ).to(device)
        checkpoint = torch.load(str(CNN_MODELS[plane]), map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        cnn_model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
        cnn_probs, _ = get_predictions(cnn_model, test_loader, device)
        all_predictions['cnn'][plane] = cnn_probs
        del cnn_model, checkpoint
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Calculate ensembles
    ensembles = {}
    for model_name in ['swin', 'vit', 'cnn']:
        # Average probability across planes
        ensembles[model_name] = (
            all_predictions[model_name]['sagittal'] + 
            all_predictions[model_name]['coronal'] + 
            all_predictions[model_name]['axial']
        ) / 3.0

    # Determine failures
    failures = {}
    for model_name in ['swin', 'vit', 'cnn']:
        probs = ensembles[model_name]
        threshold = THRESHOLDS[model_name]
        preds = (probs >= threshold).astype(int)
        
        # Misclassified indices
        failed_indices = np.where(preds != test_labels)[0]
        failures[model_name] = set(failed_indices)
        
        # Calculate metric values to verify correctness
        accuracy = np.mean(preds == test_labels)
        print(f"\n{model_name.upper()} Test Accuracy: {accuracy:.4f} (Threshold: {threshold:.4f})")
        print(f"Number of failures: {len(failed_indices)} out of {num_cases}")
        
    # Overlap analysis
    swin_fails = failures['swin']
    vit_fails = failures['vit']
    cnn_fails = failures['cnn']
    
    all_three = swin_fails.intersection(vit_fails).intersection(cnn_fails)
    swin_vit_only = swin_fails.intersection(vit_fails).difference(cnn_fails)
    swin_cnn_only = swin_fails.intersection(cnn_fails).difference(vit_fails)
    vit_cnn_only = vit_fails.intersection(cnn_fails).difference(swin_fails)
    
    swin_only = swin_fails.difference(vit_fails).difference(cnn_fails)
    vit_only = vit_fails.difference(swin_fails).difference(cnn_fails)
    cnn_only = cnn_fails.difference(swin_fails).difference(vit_fails)
    
    total_unique_failures = swin_fails.union(vit_fails).union(cnn_fails)
    
    print("\n" + "="*50)
    print("CONCORDANCE AND OVERLAP OF FAILURES")
    print("="*50)
    print(f"Total unique cases failed by at least one model: {len(total_unique_failures)}")
    print(f"Cases failed by ALL THREE models: {len(all_three)} cases")
    print(f"Cases failed by Swin and ViT (but not CNN): {len(swin_vit_only)} cases")
    print(f"Cases failed by Swin and CNN (but not ViT): {len(swin_cnn_only)} cases")
    print(f"Cases failed by ViT and CNN (but not Swin): {len(vit_cnn_only)} cases")
    print(f"Cases failed ONLY by Swin: {len(swin_only)} cases")
    print(f"Cases failed ONLY by ViT: {len(vit_only)} cases")
    print(f"Cases failed ONLY by CNN: {len(cnn_only)} cases")
    
    # Save statistics
    stats = {
        'all_three': [case_ids_list[i] for i in sorted(list(all_three))],
        'swin_vit_only': [case_ids_list[i] for i in sorted(list(swin_vit_only))],
        'swin_cnn_only': [case_ids_list[i] for i in sorted(list(swin_cnn_only))],
        'vit_cnn_only': [case_ids_list[i] for i in sorted(list(vit_cnn_only))],
        'swin_only': [case_ids_list[i] for i in sorted(list(swin_only))],
        'vit_only': [case_ids_list[i] for i in sorted(list(vit_only))],
        'cnn_only': [case_ids_list[i] for i in sorted(list(cnn_only))],
    }
    
    with open(BASE_DIR / 'scratch' / 'failures_overlap.json', 'w') as f:
        json.dump(stats, f, indent=4)
        
    print("\nSaved failure classifications to scratch/failures_overlap.json")
    
    # Plotting failure distribution
    categories = ['Swin Only', 'ViT Only', 'CNN Only', 'Swin & ViT', 'Swin & CNN', 'ViT & CNN', 'All Three']
    counts = [len(swin_only), len(vit_only), len(cnn_only), len(swin_vit_only), len(swin_cnn_only), len(vit_cnn_only), len(all_three)]
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("muted", len(categories))
    bars = plt.bar(categories, counts, color=colors, edgecolor='gray', alpha=0.9)
    plt.ylabel('Cantidad de Casos de Test', fontsize=12)
    plt.title('Distribución y Coincidencia de Fallos entre Modelos (Test Set)', fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                 f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                 
    plt.tight_layout()
    plot_path = BASE_DIR / 'scratch' / 'failure_overlap_chart.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved overlap chart to {plot_path}")
    
    # Detail table of failures
    failed_table = []
    for idx in sorted(list(total_unique_failures)):
        case_id = case_ids_list[idx]
        gt = test_labels[idx]
        swin_p = ensembles['swin'][idx]
        vit_p = ensembles['vit'][idx]
        cnn_p = ensembles['cnn'][idx]
        
        swin_pred = int(swin_p >= THRESHOLDS['swin'])
        vit_pred = int(vit_p >= THRESHOLDS['vit'])
        cnn_pred = int(cnn_p >= THRESHOLDS['cnn'])
        
        failed_table.append({
            'Case': case_id,
            'GroundTruth': int(gt),
            'SwinProb': f"{swin_p:.4f}",
            'SwinPred': swin_pred,
            'SwinFail': int(swin_pred != gt),
            'ViTProb': f"{vit_p:.4f}",
            'ViTPred': vit_pred,
            'ViTFail': int(vit_pred != gt),
            'CNNProb': f"{cnn_p:.4f}",
            'CNNPred': cnn_pred,
            'CNNFail': int(cnn_pred != gt),
        })
        
    df_failed = pd.DataFrame(failed_table)
    df_failed.to_csv(BASE_DIR / 'scratch' / 'detailed_failures.csv', index=False)
    print(f"Saved detailed failure spreadsheet to scratch/detailed_failures.csv")
    
    # Visually show failed cases where ALL THREE fail
    # We will load the Sagittal MRNet volume slice for the common failed cases and plot them side by side
    if len(all_three) > 0:
        print("\nPlotting visual representations of cases where ALL THREE models failed...")
        fig, axes = plt.subplots(len(all_three), 3, figsize=(12, 4 * len(all_three)))
        if len(all_three) == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for i, idx in enumerate(sorted(list(all_three))):
            case_id = case_ids_list[idx]
            gt_label = "Roto (LCA+)" if test_labels[idx] == 1 else "Sano (LCA-)"
            
            # Load volume from Sagittal, Coronal, Axial
            for plane_idx, plane in enumerate(PLANES):
                vol_path = DATA_DIR / 'test' / plane / f"{int(case_id):04d}.npy"
                vol = np.load(vol_path)
                
                # Get the cache indices
                cache_path = DATA_DIR / 'slice_indices_final' / f'test_{plane}_indices.json'
                with open(cache_path, 'r') as f:
                    cache = json.load(f)
                selected_indices = cache[str(int(case_id))]
                
                # We show the middle slice from the selected indices
                mid_slice_idx = selected_indices[len(selected_indices) // 2]
                slice_img = vol[mid_slice_idx]
                
                ax = axes[i, plane_idx]
                ax.imshow(slice_img, cmap='bone')
                ax.axis('off')
                if i == 0:
                    ax.set_title(plane.upper(), fontsize=12, fontweight='bold')
                
                # Add text info
                if plane_idx == 0:
                    ax.text(5, 20, f"Case: {case_id}\nGT: {gt_label}", color='yellow', 
                            bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
                    
        plt.suptitle('Cortes Centrales de Casos en los que los TRES Modelos Fallan', fontsize=14, fontweight='bold')
        plt.tight_layout()
        visual_path = BASE_DIR / 'scratch' / 'common_failures_visual.png'
        plt.savefig(visual_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved common failures visual comparison to {visual_path}")

if __name__ == '__main__':
    main()
