# Pesos de los modelos

Los pesos entrenados no se incluyen en el repositorio: ocupan del orden de 100 GB en total
(cada checkpoint multiseed son ~260 MB y hay diez semillas por plano y arquitectura). El directorio
`checkpoints/` está excluido en `.gitignore`.

## Cómo obtenerlos

Los pesos necesarios para reproducir la inferencia y las figuras de la memoria están disponibles
bajo petición al autor. <!-- TODO: sustituir por el enlace definitivo (Zenodo / Drive / Hugging Face) -->

Una vez descargados, hay que colocarlos en `checkpoints/` respetando los nombres de carpeta de
abajo, que son los que esperan los notebooks y los scripts.

## Qué hace falta para cada cosa

Para el ensamble multi-vista y la inferencia (lo que usa la memoria) bastan los checkpoints
*multiseed final* de cada arquitectura, tres por modelo (uno por plano):

```
checkpoints/
├── cnn_multiseed_resnet50/best_{sagittal,coronal,axial}_multiseed_final.pth   # ResNet50
├── vit_small_multiseed/best_{sagittal,coronal,axial}_multiseed_final.pth      # ViT-Small (22M)
├── swin_tiny_multiseed/best_{sagittal,coronal,axial}_multiseed_final.pth      # Swin-Tiny (28M)
└── acl_slice_classifier_v3/ , acl_slice_classifier_coronal/                   # selectores de cortes (MobileNetV2)
```

El selector de cortes es un MobileNetV2 con un modelo por plano (sagital y coronal); el plano axial
usa cortes centrales y no necesita selector.

## Otros checkpoints

El resto de carpetas de `checkpoints/` corresponden a experimentos de comparación que aparecen en la
memoria pero no son necesarios para la inferencia: variantes grandes (`vit_multiseed_base`,
`swin_base_baseline`), la Swin-Small auténtica que no converge (`swin_small_REAL_multiseed`),
baselines de una sola semilla (`*_baseline`) y ConvNeXt.

Aviso sobre nombres: el Swin-Tiny canónico es `swin_tiny_multiseed`. Existe `swin_tiny_multiseed_legacy`,
que es un entrenamiento anterior del mismo modelo, conservado solo por trazabilidad. Las carpetas
`swin_baseline`, `swin_tiny_baseline` y `swin_small_baseline` (renombrada a `swin_tiny_baseline`)
contienen modelos Swin-Tiny pese a algún nombre heredado; la única Swin-Small real es
`swin_small_REAL_multiseed`.

El pooling por plano debe coincidir con el del entrenamiento al cargar un checkpoint: *attention*
en sagital y axial y *max* en coronal en ResNet50 y ViT-Small, y *attention* en los tres planos en
Swin-Tiny. Los notebooks de cada arquitectura ya lo configuran.
