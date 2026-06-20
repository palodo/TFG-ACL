# Código fuente

Módulos compartidos por los notebooks y los scripts.

- `models.py` — definición de los clasificadores multi-corte para cada backbone (ResNet50, ViT en sus
  variantes, Swin, ConvNeXt) y el módulo de *attention pooling*. Cada clasificador procesa los K
  cortes seleccionados de un plano y los agrega en un único vector antes de la cabeza de clasificación.
- `data_loader.py` — `OptimizedMRNetDataset` y las transformaciones de validación/test. Lee los
  volúmenes por plano y los índices de cortes precalculados.
- `preprocessing.py` — normalización de intensidades y redimensionado de los cortes.
- `training_utils.py` — bucle de entrenamiento, *early stopping* y métricas.
- `gpu_config.py` — configuración de dispositivo.

Una nota sobre `ViTSmallMultiSliceClassifier`: con `pretrained=False` construye la configuración por
defecto de ViT (dimensión 768). Para cargar los pesos de ViT-Small (dimensión 384) hay que
instanciarlo con `pretrained=True`, que toma la configuración correcta del modelo de Hugging Face.
