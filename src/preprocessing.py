"""
Preprocessing module for ACL Classifier
Handles slice index calculation and caching.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch


def load_volume_normalized(volume_path):
    """
    Load a volume .npy and normalize it to [0, 1].
    
    Args:
        volume_path: Path to the .npy file
        
    Returns:
        numpy array normalized to [0, 1] with dtype float32
    """
    volume = np.load(volume_path)
    
    # Min-max normalization to [0, 1]
    volume_min = volume.min()
    volume_max = volume.max()
    if volume_max > volume_min:
        volume = (volume - volume_min) / (volume_max - volume_min)
    
    return volume.astype(np.float32)


def compute_center_consecutive_indices(num_slices_total, k):
    """
    Compute indices of k center consecutive slices.
    Used for axial plane strategy.
    
    Args:
        num_slices_total: Total number of slices in volume
        k: Number of slices to select
        
    Returns:
        list of k consecutive indices centered around middle
    """
    center = num_slices_total // 2
    start = max(0, center - k // 2)
    end = min(num_slices_total, start + k)
    
    # Adjust if we go past the end
    if end - start < k:
        start = max(0, end - k)
    
    return list(range(start, start + k))


def preprocess_slice_indices(data_path, slice_indices_dir, plane_config, cnn_selectors, 
                             use_cached_indices=True, splits=['train', 'val', 'test']):
    """
    Main preprocessing function for multi-plane CNN-based and center-based slice selection.
    
    For each split and plane:
    - Sagittal/Coronal: Uses CNN Selector to choose K best slices
    - Axial: Uses center consecutive slices
    
    Saves results to JSON files for reuse.
    
    Args:
        data_path: Path to data directory (contains train/, val/, test/)
        slice_indices_dir: Path where to save JSON files with computed indices
        plane_config: Dict with config for each plane (num_slices, strategy, etc.)
        cnn_selectors: Dict with loaded SimpleCNNSelector objects (or None for non-CNN planes)
        use_cached_indices: If True, skip planes that already have saved JSON files
        splits: List of data splits to process ('train', 'val', 'test')
    """
    print(f"{'='*80}")
    print(f"PRE-PROCESSING: CÁLCULO DE ÍNDICES DE SLICES (MULTI-CNN)")
    print(f"{'='*80}")
    print(f"\nEstrategia por plano:")
    print(f"  • Sagittal: K={plane_config['sagittal']['num_slices']} (CNN Selector V3 - GPU acelerada)")
    print(f"  • Coronal: K={plane_config['coronal']['num_slices']} (CNN Selector Coronal - GPU acelerada)")
    print(f"  • Axial: K={plane_config['axial']['num_slices']} (slices centrales consecutivos)")
    print(f"\n{'='*80}\n")
    
    planes = list(plane_config.keys())
    data_path = Path(data_path)
    slice_indices_dir = Path(slice_indices_dir)
    slice_indices_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in splits:
        print(f"\n{'='*70}")
        print(f"PROCESANDO SPLIT: {split.upper()}")
        print(f"{'='*70}\n")
        
        # Process each plane
        for plane in planes:
            config = plane_config[plane]
            k = config['num_slices']
            strategy = config['strategy']
            
            # Path where to save indices
            output_path = slice_indices_dir / f"{split}_{plane}_indices.json"
            
            # If already cached and using cache, skip
            if use_cached_indices and output_path.exists():
                print(f"[{plane.upper()}] ✓ Usando índices cacheados: {output_path.name}")
                continue
            
            print(f"[{plane.upper()}] Calculando índices (K={k}, {strategy})...")
            
            # Load CSV
            if split == 'val':
                csv_name = 'val-acl.csv'
            else:
                csv_name = f'{split}-acl.csv'
            
            csv_path = data_path / csv_name
            df = pd.read_csv(csv_path, header=None, names=['case', 'label'])
            
            # Dict to save: {case_id: [indices]}
            indices_dict = {}
            
            # Process according to strategy
            if strategy == 'cnn_based' and plane in cnn_selectors and cnn_selectors[plane] is not None:
                # CNN-based strategy
                cnn_selector = cnn_selectors[plane]
                print(f"   Procesando {len(df)} casos con CNN Selector ({plane})...")
                
                total_cases = len(df)
                print_every = max(1, total_cases // 10)
                
                for idx, (_, row) in enumerate(df.iterrows()):
                    case_id = row['case']
                    volume_path = data_path / split / plane / f"{case_id:04d}.npy"
                    volume = load_volume_normalized(volume_path)
                    num_slices = volume.shape[0]
                    
                    # Compute ACL probability for each slice
                    acl_probs = []
                    for slice_idx in range(num_slices):
                        prob = cnn_selector.get_acl_probability(volume[slice_idx])
                        acl_probs.append(prob)
                    
                    # Select K slices with highest probability
                    indices = np.argsort(acl_probs)[-k:]
                    indices = sorted(indices.tolist())
                    indices_dict[str(case_id)] = indices
                    
                    # Progress
                    if (idx + 1) % print_every == 0 or (idx + 1) == total_cases:
                        progress = (idx + 1) / total_cases * 100
                        print(f"      Progreso: {idx+1}/{total_cases} ({progress:.0f}%)")
            
            elif strategy == 'center_consecutive':
                # Center-based strategy (fast, no GPU needed)
                print(f"   Calculando índices centrales para {len(df)} casos...")
                
                for _, row in df.iterrows():
                    case_id = row['case']
                    volume_path = data_path / split / plane / f"{case_id:04d}.npy"
                    volume = np.load(volume_path)
                    num_slices_total = volume.shape[0]
                    
                    # Compute center indices
                    indices = compute_center_consecutive_indices(num_slices_total, k)
                    indices_dict[str(case_id)] = indices
            
            # Save indices to JSON
            with open(output_path, 'w') as f:
                json.dump(indices_dict, f, indent=2)
            
            print(f"   ✓ Guardado: {output_path.name} ({len(indices_dict)} casos)\n")
    
    print(f"\n{'='*80}")
    print(f"✓ PRE-PROCESAMIENTO COMPLETADO")
    print(f"{'='*80}\n")
