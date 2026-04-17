# Vision Transformer Small Baseline - ACL Classifier

## Overview

Este directorio contiene la implementación del entrenamiento de un modelo **Vision Transformer Small** para la clasificación de lesiones de ACL en resonancias magnéticas.

## Key Differences from ViT Tiny

| Aspecto | ViT Tiny | ViT Small |
|--------|---------|-----------|
| **Modelo HuggingFace** | `WinKawaks/vit-tiny-patch16-224` | `WinKawaks/vit-small-patch16-224` |
| **Parámetros** | ~5.7M | ~22M |
| **Dimensión de features** | 192D | 768D |
| **Learning Rate** | 1.5e-4 | 1.2e-4 |
| **Weight Decay (Sagittal)** | 5e-4 | 1e-3 |
| **Weight Decay (Axial)** | 2e-4 | 5e-4 |
| **Tamaño del modelo** | Ultra-compacto | Compacto |
| **Velocidad de inferencia** | Más rápida | Más lenta que Tiny, más rápida que Base |

## Structure

- **vit_small_baseline.ipynb**: Notebook principal con pipeline completo de entrenamiento
  - Imports y configuración
  - Entrenamiento por plano anatómico (Sagittal, Coronal, Axial)
  - Visualización de resultados por plano
  - Carga de modelos para ensamble
  - Predicciones en validación y test
  - Calibración de threshold
  - Resultados finales

## Rutas (3 niveles de profundidad)

Dado que el notebook está en `models/vit_small/`, todas las rutas usan:
- `../../../data` → `/data`
- `../../../checkpoints/vit_small_baseline` → `/checkpoints/vit_small_baseline`
- `../../../src` → `/src`

## Model Class

La clase `ViTSmallMultiSliceClassifier` se encuentra en `src/models.py` y proporciona:
- Carga de ViT Small preentrenado desde HuggingFace
- Processing de múltiples slices MRI
- Attention pooling y classificación
- Support para different pooling modes (attention, max, mean)

## Configuration

La configuración por plano está definida en el notebook:

### Sagittal
- Learning Rate: 1.2e-4
- Weight Decay: 1e-3
- Dropout: 0.2 / 0.15
- Early Stopping Patience: 25
- Warmup Epochs: 5

### Coronal
- Learning Rate: 1.2e-4
- Weight Decay: 1e-3
- Dropout: 0.25 / 0.15
- Grad Clip: ✓
- Early Stopping Patience: 15
- Warmup Epochs: 5

### Axial
- Learning Rate: 1.2e-4
- Weight Decay: 5e-4
- Dropout: 0.25 / 0.15
- Grad Clip: ✓
- Label Smoothing: 0.05
- Early Stopping Patience: 15
- Warmup Epochs: 5

## Checkpoints

Los checkpoints se guardan en `/checkpoints/vit_small_baseline/`:
- `best_sagittal_auc.pth`
- `best_coronal_auc.pth`
- `best_axial_auc.pth`
- `ensemble_results_calibrated.json`
- `ensemble_roc_curves.png`

## Usage

1. Abre el notebook `vit_small_baseline.ipynb` en Jupyter
2. Ejecuta cell 1: Imports
3. Ejecuta cell 2: Global Configuration
4. Ejecuta cells 3-4: Entrenar Sagittal
5. Ejecuta cells 5-6: Entrenar Coronal  
6. Ejecuta cells 7-8: Entrenar Axial
7. Ejecuta cell 9: Load ensemble models
8. Ejecuta cell 10: Make predictions
9. Ejecuta cells 11-12: Threshold calibration & final results

## Comparación con otros baselines

- **ViT Tiny**: 5.7M params, 192D features, más rápido, menos capacidad
- **ViT Small**: 22M params, 768D features, balance eficiencia-performance (ESTE)
- **ViT Base**: 86M params, 768D features, más potente, más lento
- **ConvNeXt**: 27.8M params, 768D features, similar performance
- **Swin**: ~28M params, 768D features, similar performance

## Notes

- El modelo requiere importar `ViTSmallMultiSliceClassifier` de `src.models`
- Los datos deben estar en la estructura estándar: `data/train/`, `data/val/`, `data/test/`
- Los slice indices deben estar precalculados en `data/slice_indices_final/`
- GPU allocation: GPUs 1,2,3 (GPU 0 oculta para evitar problemas)
