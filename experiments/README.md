# Modelos Disponibles

## 📙 Descripción de Notebooks

### 1. ConvNeXt V2 Tiny (`convnext/convnext_baseline.ipynb`)

**Descripción:** Pipeline de entrenamiento para ConvNeXt V2 Tiny, arquitectura moderna basada en ConvMixer que mejora sobre ResNet.

**Características:**
- Backbone: `convnext_tiny_fcmae.in1k_ft_in1k` (27.8M parámetros)
- Feature dimension: 768D
- Pooling strategies: Attention, Max, Mean
- Augmentación agresiva en 3 planos

**Resultados finales:**
- Sagittal: AUC 0.9719, F1 0.8060
- Coronal: AUC 0.9134, F1 0.7442  
- Axial: AUC 0.9695, F1 0.7423
- **Ensemble Val: AUC 0.9790**
- **Ensemble Test: AUC 0.9251**

**Uso:**
```python
DATA_PATH = Path('../../data')
CHECKPOINT_DIR = Path('../../checkpoints/convnext_baseline_v2_small')
# El notebook carga automáticamente desde estas rutas
```

---

### 2. Swin Transformer (`swin/swin_baseline.ipynb`)

**Descripción:** Pipeline de entrenamiento para Swin Transformer Tiny, arquitectura basada en attention que es competitive con ConvNeXt.

**Características:**
- Backbone: `swin_tiny_patch4_window7_224.ms_in1k` (28M parámetros)
- Feature dimension: 768D
- Identical kernel config a ConvNeXt para comparación justa
- Threshold strategy: MAX RECALL with Precision >= 0.75

**Resultados esperados:**
- Comparable a ConvNeXt (AUC ~0.97+ ensemble)

**Uso:**
```python
DATA_PATH = Path('../../data')
CHECKPOINT_DIR = Path('../../checkpoints/swin_baseline')
```

---

### 3. ResNet50 CNN (`cnn/cnn_baseline.ipynb`)

**Descripción:** Baseline con ResNet50 pre-entrenado para referencia de performance.

**Características:**
- Backbone: `resnet50` (23M parámetros)
- Feature dimension: 2048D → proyectado a 768D
- Regularización más fuerte (dropout 42-47%)
- LR Annealing: ReduceLROnPlateau

**Resultados:**
- Ensemble Val: ~0.95+ AUC
- Ensemble Test: ~0.91+ AUC

**Uso:**
```python
DATA_PATH = Path('../../data')
CHECKPOINT_DIR = Path('../../checkpoints/cnn_baseline_resnet50')
```

---

### 4. Vision Transformer (ViT) (`vit/`)

**Descripción:** Implementación completa del pipeline ViT con múltiples notebooks.

**Archivos:**
- `train.ipynb` - Entrenamiento principal
- `ensemble_evaluation.ipynb` - Evaluación de ensamble
- `attention_maps_visualization.ipynb` - Visualización de mapas de atención

**Características:**
- Backbone: ViT-B (86M parámetros)
- Entrenamiento multi-plano independiente
- Early stopping adaptativo

**Resultados:**
- Ensemble Val: ~0.96+ AUC
- Ensemble Test: ~0.92+ AUC

**Uso:**
```python
DATA_PATH = Path('../../data')
CHECKPOINT_DIR = Path('../../checkpoints/vit_optimized_pipeline')
```

---

## 🔄 Flujo de ejecución (cualquier notebook)

1. **Configuración global** - Rutas, hiperparámetros por plano
2. **Entrenamiento sagital** - Entrena modelo + visualiza curvas
3. **Entrenamiento coronal** - Ídem
4. **Entrenamiento axial** - Ídem
5. **Cargar modelos para ensamble** - Carga los 3 checkpoints
6. **Predicciones** - Validación + Test
7. **Calibración de threshold** - 3 opciones (MAX F1, MAX RECALL, MAX YOUDENS)
8. **Metricas finales** - Confusion matrices, ROC curves, resumen

## 📊 Comparación de modelos

| Métrica | ConvNeXt | Swin | ResNet50 | ViT |
|---------|----------|------|----------|-----|
| Ensemble AUC (Val) | **0.9790** | ~0.97 | ~0.95 | ~0.96 |
| Ensemble AUC (Test) | **0.9251** | ~0.92 | ~0.91 | ~0.92 |
| Parámetros | 27.8M | 28M | 23M | 86M |
| Velocidad | Rápido | Medio | Muy rápido | Lento |

## 🎯 Recomendación

**Para producción:** Usar **ConvNeXt V2 Tiny** (mejor AUC, pequeño tamaño, rápido)

**Para análisis:** Usar **Swin Transformer** (comparable a ConvNeXt, arquitectura distinta)

**Para referencia:** **ResNet50** (más pequeño, más rápido, AUC aceptable)
