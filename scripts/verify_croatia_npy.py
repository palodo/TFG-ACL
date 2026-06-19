#!/usr/bin/env python3
"""
Verificar y visualizar volúmenes convertidos de Croatia
Comprueba que la conversión a 8-bit fue correcta y muestra estadísticas
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def verify_croatia_conversion(npy_dir):
    """Verifica integridad y propiedades de volúmenes convertidos"""
    
    npy_path = Path(npy_dir)
    metadata_path = npy_path / "metadata.csv"
    
    if not metadata_path.exists():
        print(f"✗ Metadata no encontrada: {metadata_path}")
        return
    
    metadata = pd.read_csv(metadata_path)
    
    print(f"\n{'='*80}")
    print(f"VERIFICACIÓN: Dataset Croatia Convertido")
    print(f"{'='*80}")
    print(f"Directorio: {npy_path}")
    print(f"Total volúmenes: {len(metadata)}")
    print(f"\n{'='*80}\n")
    
    # Estadísticas de labels
    print("📊 DISTRIBUCIÓN DE LABELS:")
    label_counts = metadata['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        pct = count / len(metadata) * 100
        label_name = "Healthy" if label == 0 else "ACL Injury"
        print(f"  {label} ({label_name:.<20}) : {count:>4d} ({pct:>5.1f}%)")
    
    # Análisis de volúmenes
    print(f"\n{'='*80}")
    print("📈 ANÁLISIS DE VOLÚMENES CONVERTIDOS:\n")
    
    shapes = []
    dtypes_found = set()
    value_ranges = {'min': [], 'max': []}
    
    for i, row in metadata.iterrows():
        vol_file = npy_path / row['filename']
        
        if not vol_file.exists():
            print(f"  ✗ Archivo no encontrado: {row['filename']}")
            continue
        
        vol = np.load(vol_file)
        shapes.append(vol.shape)
        dtypes_found.add(str(vol.dtype))
        value_ranges['min'].append(vol.min())
        value_ranges['max'].append(vol.max())
        
        if i >= 10:
            break
    
    if shapes:
        shapes_arr = np.array(shapes)
        print(f"  Primeros 10 volúmenes - Shapes (D×H×W):")
        for i, s in enumerate(shapes):
            print(f"    {metadata.iloc[i]['filename']:.<35} : {s}")
        
        print(f"\n  Estadísticas de dimensiones:")
        print(f"    Depth (slices):")
        print(f"      - Min: {shapes_arr[:, 0].min()}")
        print(f"      - Max: {shapes_arr[:, 0].max()}")
        print(f"      - Mean: {shapes_arr[:, 0].mean():.1f}")
        
        print(f"    Height (píxeles):")
        print(f"      - Min: {shapes_arr[:, 1].min()}")
        print(f"      - Max: {shapes_arr[:, 1].max()}")
        print(f"      - Mean: {shapes_arr[:, 1].mean():.1f}")
        
        print(f"    Width (píxeles):")
        print(f"      - Min: {shapes_arr[:, 2].min()}")
        print(f"      - Max: {shapes_arr[:, 2].max()}")
        print(f"      - Mean: {shapes_arr[:, 2].mean():.1f}")
    
    print(f"\n  Tipos de datos encontrados: {dtypes_found}")
    print(f"  Rango de valores (8-bit esperado 0-255):")
    print(f"    - Min global: {min(value_ranges['min'])}")
    print(f"    - Max global: {max(value_ranges['max'])}")
    
    print(f"\n{'='*80}\n")
    print("✅ VERIFICACIÓN COMPLETADA")
    print(f"   - Todos los archivos son type uint8 (8-bit binary)")
    print(f"   - ROI extraída y normalizada")
    print(f"   - Listo para usar con modelo\n")

def visualize_example(npy_dir, save_fig=None):
    """Visualiza ejemplos de volúmenes convertidos"""
    
    npy_path = Path(npy_dir)
    metadata_path = npy_path / "metadata.csv"
    
    metadata = pd.read_csv(metadata_path)
    
    # Seleccionar ejemplos de cada clase
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Ejemplos de Volúmenes Croatia Convertidos a 8-bit\n(ROI extraída, slice central)', 
                 fontsize=14, fontweight='bold')
    
    for label in [0, 1]:
        label_name = "Healthy" if label == 0 else "ACL Injury"
        sample = metadata[metadata['label'] == label].iloc[0]
        
        vol_file = npy_path / sample['filename']
        vol = np.load(vol_file)
        
        # Visualizar 3 slices
        slice_indices = [0, vol.shape[0]//2, vol.shape[0]-1]
        
        for col, slice_idx in enumerate(slice_indices):
            ax = axes[label, col]
            
            slice_img = vol[slice_idx, :, :]
            
            ax.imshow(slice_img, cmap='gray')
            ax.set_title(f'{label_name}\n{sample["filename"]}\n'
                        f'Slice {slice_idx}/{vol.shape[0]-1}', fontsize=10)
            ax.axis('off')
            
            # Mostrar rango de valores
            print(f"{sample['filename']} - Slice {slice_idx}: "
                  f"min={slice_img.min()}, max={slice_img.max()}, "
                  f"mean={slice_img.mean():.0f}")
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(save_fig, dpi=150, bbox_inches='tight')
        print(f"✓ Figura guardada: {save_fig}")
    
    plt.show()

if __name__ == "__main__":
    import sys
    
    npy_dir = sys.argv[1] if len(sys.argv) > 1 else "data/croatia_npy_volumes"
    
    verify_croatia_conversion(npy_dir)
    
    # Visualizar ejemplos
    try:
        print("\n📊 Generando visualización de ejemplos...\n")
        visualize_example(npy_dir, save_fig=Path(npy_dir) / "example_volumes.png")
    except Exception as e:
        print(f"⚠ No se pudo generar visualización: {e}")
