import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_full_case_plot(case_id, data_dir, output_dir):
    planes = ['sagittal', 'coronal', 'axial']
    
    # Load index caches
    caches = {}
    for plane in planes:
        cache_path = data_dir / 'slice_indices_final' / f'test_{plane}_indices.json'
        with open(cache_path, 'r') as f:
            caches[plane] = json.load(f)
            
    fig, axes = plt.subplots(3, 10, figsize=(20, 8.5))
    
    plane_display_names = {
        'sagittal': 'SAGITAL\n(5 cortes)',
        'coronal': 'CORONAL\n(10 cortes)',
        'axial': 'AXIAL\n(10 cortes)'
    }
    
    for row_idx, plane in enumerate(planes):
        # Load volume
        vol_path = data_dir / 'test' / plane / f"{int(case_id):04d}.npy"
        vol = np.load(vol_path)
        
        # Get selected slice indices
        selected_indices = caches[plane][str(int(case_id))]
        num_slices = len(selected_indices)
        
        # Plot slices
        for col_idx in range(10):
            ax = axes[row_idx, col_idx]
            
            # Hide spines and ticks but keep axes enabled so y-label works
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            
            if col_idx < num_slices:
                slice_idx = selected_indices[col_idx]
                slice_img = vol[slice_idx]
                ax.imshow(slice_img, cmap='bone')
                ax.set_title(f"Corte {slice_idx}", fontsize=9, pad=4)
            else:
                # Hide empty cells completely
                ax.axis('off')
                
            # Row label on the first column (placed to the left, horizontally)
            if col_idx == 0:
                ax.set_ylabel(plane_display_names[plane], fontsize=12, fontweight='bold', 
                              labelpad=25, rotation=0, va='center', ha='right')
                
    fig.suptitle(f"Caso {case_id} (Falso Negativo - Real: LCA Roto, Pred: Sano)", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust spacing to avoid overlap and leave space on the left for the label
    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.12, right=0.98, hspace=0.35, wspace=0.15)
    
    output_path = output_dir / f"case_{case_id}_all_slices.png"
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")

def main():
    BASE_DIR = Path('/home/palodo2/acl_classifier')
    DATA_DIR = BASE_DIR / 'data'
    OUTPUT_DIR = BASE_DIR / 'scratch'
    
    cases = ["0080", "0087", "0521"]
    for case in cases:
        generate_full_case_plot(case, DATA_DIR, OUTPUT_DIR)

if __name__ == '__main__':
    main()
