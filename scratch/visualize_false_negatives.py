import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    BASE_DIR = Path('/home/palodo2/acl_classifier')
    DATA_DIR = BASE_DIR / 'data'
    
    # The 3 cases where all models got a False Negative (GT=1, Predicted=0)
    cases = ["0080", "0087", "0521"]
    
    # Load sagittal indices cache
    sag_cache_path = DATA_DIR / 'slice_indices_final' / 'test_sagittal_indices.json'
    with open(sag_cache_path, 'r') as f:
        sag_cache = json.load(f)
        
    fig, axes = plt.subplots(len(cases), 5, figsize=(15, 3.5 * len(cases)))
    
    for row_idx, case_id in enumerate(cases):
        # Load sagittal volume
        vol_path = DATA_DIR / 'test' / 'sagittal' / f"{int(case_id):04d}.npy"
        vol = np.load(vol_path)
        
        # Get selected slice indices
        selected_indices = sag_cache[str(int(case_id))]
        
        # Plot all 5 slices for the case
        for col_idx, slice_idx in enumerate(selected_indices):
            ax = axes[row_idx, col_idx]
            
            # Show slice
            slice_img = vol[slice_idx]
            ax.imshow(slice_img, cmap='bone')
            ax.axis('off')
            
            # Add titles and labels
            ax.set_title(f"Corte {slice_idx}", fontsize=10)
            
            # Label on the first slice of the row
            if col_idx == 0:
                ax.text(-40, 112, f"Caso {case_id}\n(FN LCA+)", color='red', fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.8, boxstyle='round,pad=0.4'))
                
    plt.suptitle("Cortes Sagitales Seleccionados de los 3 Falsos Negativos Comunes (GT=1, Pred=0)", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = BASE_DIR / 'scratch' / 'false_negatives_all_three.png'
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated visual plot: {output_path}")

if __name__ == '__main__':
    main()
