#!/usr/bin/env python3
"""
Convertir Dataset Croatia de pickle + ROI annotations a .npy volumes
Normaliza de 12-bit a 8-bit (binario) y guarda en estructura similar a MRNet

Uso:
    python3 convert_croatia_to_npy.py --input data/Croatia --output data/croatia_npy_volumes
"""

import os
import sys
import pickle
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_volume_from_pickle(pck_path):
    """Carga volumen 3D desde archivo pickle"""
    with open(pck_path, 'rb') as f:
        volume = pickle.load(f)
    return volume

def normalize_to_8bit(volume):
    """
    Normaliza volumen de 12-bit (0-4095) a 8-bit (0-255)
    Asume rango de entrada [0-4095]
    """
    # Asegurarse que es float para operaciones
    volume = volume.astype(np.float32)
    
    # Obtener valores min y max reales
    v_min = volume.min()
    v_max = volume.max()
    
    # Normalizar a [0, 1]
    if v_max > v_min:
        volume_normalized = (volume - v_min) / (v_max - v_min)
    else:
        volume_normalized = volume
    
    # Escalar a [0, 255] y convertir a uint8
    volume_8bit = (volume_normalized * 255).astype(np.uint8)
    
    return volume_8bit

def extract_roi_volume(volume, roi_info):
    """
    Extrae la región de interés del volumen
    
    Args:
        volume: volumen 3D completo
        roi_info: dict con roiX, roiY, roiZ, roiWidth, roiHeight, roiDepth
    
    Returns:
        roi_volume: volumen extraído (depth, height, width)
    """
    z_start = roi_info['roiZ']
    z_end = z_start + roi_info['roiDepth']
    
    x_start = roi_info['roiX']
    x_end = x_start + roi_info['roiWidth']
    
    y_start = roi_info['roiY']
    y_end = y_start + roi_info['roiHeight']
    
    # Extraer ROI: [depth, height, width]
    roi_volume = volume[z_start:z_end, y_start:y_end, x_start:x_end]
    
    return roi_volume

def process_croatia_dataset(input_dir, output_dir, use_roi=True, verbose=False):
    """
    Procesa todo el dataset de Croatia
    
    Args:
        input_dir: path a data/Croatia
        output_dir: path donde guardar volumes .npy
        use_roi: si True, extrae ROI; si False, guarda volumen completo
        verbose: mostrar información detallada
    
    Returns:
        dict con estadísticas del procesamiento
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Verificar directorios
    if not input_path.exists():
        raise FileNotFoundError(f"Input dir not found: {input_path}")
    
    volumetric_dir = input_path / "volumetric_data"
    metadata_path = input_path / "metadata.csv"
    
    if not volumetric_dir.exists():
        raise FileNotFoundError(f"Volumetric data dir not found: {volumetric_dir}")
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Crear directorio de salida
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Cargar metadata
    metadata = pd.read_csv(metadata_path)
    
    # Mapear diagnósticos a etiquetas binarias
    # 0 = Healthy (label 0)
    # 1,2 = Injured (label 1)
    metadata['label'] = (metadata['aclDiagnosis'] > 0).astype(int)
    metadata['filename'] = metadata['volumeFilename'].str.replace('.pck', '.npy')
    
    print(f"\n{'='*80}")
    print(f"CONVERTINDO DATASET CROATIA A BINARIO 8-BIT (.npy)")
    print(f"{'='*80}")
    print(f"Input dir: {input_path}")
    print(f"Output dir: {output_path}")
    print(f"Volumetric files: {volumetric_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Total casos: {len(metadata)}")
    print(f"Usar ROI: {use_roi}")
    print(f"{'='*80}\n")
    
    # Procesar cada volumen
    stats = {
        'total': len(metadata),
        'success': 0,
        'errors': 0,
        'volumes_info': []
    }
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Procesando"):
        try:
            pck_filename = row['volumeFilename']
            npy_filename = row['filename']
            
            pck_path = volumetric_dir / pck_filename
            output_file = output_path / npy_filename
            
            # Verificar que archivo existe
            if not pck_path.exists():
                print(f"  ✗ No encontrado: {pck_filename}")
                stats['errors'] += 1
                continue
            
            # Cargar volumen
            volume = load_volume_from_pickle(pck_path)
            
            # Extraer ROI si está indicado
            if use_roi:
                volume = extract_roi_volume(volume, row)
            
            # Normalizar a 8-bit
            volume_8bit = normalize_to_8bit(volume)
            
            # Guardar como .npy
            np.save(str(output_file), volume_8bit)
            
            # Registrar información
            stats['volumes_info'].append({
                'filename': npy_filename,
                'original_shape': str(volume.shape),
                'normalized_shape': str(volume_8bit.shape),
                'label': row['label'],
                'diagnosis': row['aclDiagnosis'],
                'acl_diagnosis_name': {0: 'Healthy', 1: 'Partially Injured', 2: 'Completely Ruptured'}.get(row['aclDiagnosis'], 'Unknown')
            })
            
            stats['success'] += 1
            
            if verbose and (idx + 1) % 100 == 0:
                print(f"  Procesados {idx+1}/{len(metadata)} volúmenes")
        
        except Exception as e:
            print(f"  ✗ Error procesando {pck_filename}: {str(e)[:60]}")
            stats['errors'] += 1
            continue
    
    print(f"\n{'='*80}")
    print(f"RESULTADO:")
    print(f"  ✓ Éxito: {stats['success']}/{stats['total']}")
    print(f"  ✗ Errores: {stats['errors']}/{stats['total']}")
    print(f"{'='*80}\n")
    
    # Guardar CSV con metadata actualizada
    output_csv = output_path / "metadata.csv"
    metadata_output = metadata.copy()
    metadata_output = metadata_output[['filename', 'label', 'aclDiagnosis', 'kneeLR', 'roiX', 'roiY', 'roiZ', 'roiWidth', 'roiHeight', 'roiDepth']]
    metadata_output.to_csv(output_csv, index=False)
    print(f"✓ Metadata guardada: {output_csv}\n")
    
    # Guardar estadísticas
    stats_json = output_path / "conversion_stats.json"
    with open(stats_json, 'w') as f:
        # Convertir a JSON serializable
        stats_serializable = {
            'total': stats['total'],
            'success': stats['success'],
            'errors': stats['errors'],
            'volumes_info': stats['volumes_info']
        }
        json.dump(stats_serializable, f, indent=2)
    
    print(f"✓ Estadísticas guardadas: {stats_json}\n")
    
    return stats

def main():
    parser = argparse.ArgumentParser(
        description="Convertir dataset Croatia de pickle a .npy normalizado (8-bit)"
    )
    parser.add_argument('--input', default='data/Croatia', 
                       help='Path a directorio Croatia (default: data/Croatia)')
    parser.add_argument('--output', default='data/croatia_npy_volumes',
                       help='Path output para .npy files (default: data/croatia_npy_volumes)')
    parser.add_argument('--no-roi', action='store_true',
                       help='Guardar volumen completo sin extraer ROI')
    parser.add_argument('--verbose', action='store_true',
                       help='Mostrar información detallada')
    
    args = parser.parse_args()
    
    try:
        stats = process_croatia_dataset(
            args.input, 
            args.output,
            use_roi=not args.no_roi,
            verbose=args.verbose
        )
        
        print("\n✅ Conversión completada exitosamente!")
        print(f"   Archivos guardados en: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
