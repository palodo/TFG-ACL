"""
Script para generar índices de slices para dataset de Croacia usando CNN Selector.
Usa el mismo modelo CNN Selector V3 que se usó para validación/test.
Solo genera índices para sagittal (K=5).
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm

# Configuración
DATA_DIR = Path('/home/palodo2/tfg/acl_classifier/data')
CROATIA_DIR = DATA_DIR / 'croatia_npy_volumes_final'
CROATIA_METADATA_CSV = CROATIA_DIR / 'metadata_final.csv'
OUTPUT_DIR = DATA_DIR / 'slice_indices_final'
CNN_SELECTOR_PATH = Path('/home/palodo2/tfg/acl_classifier/checkpoints/acl_slice_classifier_v3/best_model_f1.pth')

OUTPUT_FILE = OUTPUT_DIR / 'croatia_sagittal_indices.json'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
K = 5  # Número de slices a seleccionar


class CNNSelectorV3(nn.Module):
    """Arquitectura del CNN Selector V3 - identifica slices con ACL"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  # 2 clases: con/sin ACL
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class SimpleCNNSelector:
    """Wrapper del CNN Selector para producción"""
    def __init__(self, checkpoint_path, device='cuda:0'):
        self.device = device
        self.model = CNNSelectorV3().to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"CNN Selector cargado desde: {checkpoint_path}")
    
    def get_acl_probability(self, slice_2d):
        """
        Input: slice 2D (H, W) normalizado a [0, 1]
        Output: probabilidad de contener ACL
        """
        # Asegurar que es numpy
        if isinstance(slice_2d, torch.Tensor):
            slice_2d = slice_2d.cpu().numpy()
        
        # Convertir a tensor y añadir dimensiones
        slice_tensor = torch.from_numpy(slice_2d).float().unsqueeze(0).unsqueeze(0)
        slice_tensor = slice_tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(slice_tensor)
            probs = torch.softmax(logits, dim=1)
            acl_prob = probs[0, 1].item()  # Probabilidad clase 1 (ACL)
        
        return acl_prob


def load_volume_normalized(volume_path):
    """Cargar volumen normalizado a [0, 1]"""
    volume = np.load(volume_path).astype(np.float32)
    vol_min = volume.min()
    vol_max = volume.max()
    if vol_max > vol_min:
        volume = (volume - vol_min) / (vol_max - vol_min)
    return volume


def generate_croatia_indices():
    """Generar índices de slices para Croacia usando CNN Selector"""
    
    print("="*80)
    print("GENERANDO INDICES DE SLICES PARA CROACIA (CNN SELECTOR)")
    print("="*80)
    
    # Verificar que existe el CSV de metadatos
    if not CROATIA_METADATA_CSV.exists():
        print(f"Error: Metadata CSV no encontrado en {CROATIA_METADATA_CSV}")
        return
    
    # Cargar CNN Selector
    if not CNN_SELECTOR_PATH.exists():
        print(f"Error: CNN Selector no encontrado en {CNN_SELECTOR_PATH}")
        return
    
    print(f"\nCargando CNN Selector desde: {CNN_SELECTOR_PATH}")
    cnn_selector = SimpleCNNSelector(CNN_SELECTOR_PATH, device=DEVICE)
    
    # Cargar metadatos de Croacia
    print(f"\nCargando metadatos de Croacia...")
    croatia_df = pd.read_csv(CROATIA_METADATA_CSV)
    print(f"Volúmenes a procesar: {len(croatia_df)}")
    
    # Generar índices
    print(f"\nProcesando {len(croatia_df)} volúmenes con CNN Selector (K={K})...")
    indices_dict = {}
    
    for idx, (_, row) in enumerate(tqdm(croatia_df.iterrows(), total=len(croatia_df), desc="Generando índices")):
        volume_id = row['volume_id']
        volume_path = CROATIA_DIR / f"{volume_id:04d}.npy"
        
        if not volume_path.exists():
            print(f"Advertencia: Volumen no encontrado: {volume_path}")
            continue
        
        # Cargar volumen normalizado
        volume = load_volume_normalized(volume_path)
        num_slices = volume.shape[0]
        
        # Calcular probabilidad ACL para cada slice
        acl_probs = []
        for slice_idx in range(num_slices):
            slice_2d = volume[slice_idx]
            prob = cnn_selector.get_acl_probability(slice_2d)
            acl_probs.append(prob)
        
        # Seleccionar K slices con mayor probabilidad
        acl_probs = np.array(acl_probs)
        if num_slices <= K:
            # Si hay pocos slices, usar todos
            selected_indices = list(range(num_slices))
            # Rellenar con repetición si necesario
            while len(selected_indices) < K:
                selected_indices.append(selected_indices[-1])
            selected_indices = selected_indices[:K]
        else:
            # Seleccionar los K mejores
            selected_indices = np.argsort(acl_probs)[-K:].tolist()
            selected_indices = sorted(selected_indices)
        
        indices_dict[str(volume_id)] = selected_indices
    
    # Guardar índices
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(indices_dict, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Índices guardados: {OUTPUT_FILE}")
    print(f"Volúmenes procesados: {len(indices_dict)}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    generate_croatia_indices()
