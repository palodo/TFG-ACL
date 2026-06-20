# Swin-Tiny

Entrenamiento y evaluación de Swin-Tiny (`swin_tiny_patch4_window7_224`, ~28 M de parámetros), la
arquitectura de atención jerárquica del trabajo y una de las tres principales. Usa *attention
pooling* en los tres planos (a diferencia de ResNet50 y ViT-Small, que usan *max* en coronal).

## Notebooks

- `swin_tiny_multiseed.ipynb` — entrenamiento multiseed que produce el modelo **canónico** del
  trabajo, en `checkpoints/swin_tiny_multiseed/`. Es el que usan la memoria, la app y el resto de
  evaluaciones.
- `swin_tiny_multiseed_legacy.ipynb` — un entrenamiento multiseed **anterior** del mismo modelo
  (`checkpoints/swin_tiny_multiseed_legacy/`). Se conserva por trazabilidad, pero no es el que se usa
  en los resultados.
- `Threshold_Tuning_and_External_Validation.ipynb` — ensamble de los tres planos del modelo canónico,
  umbral clínico sobre validación (recall máximo con precisión >= 0,75), test de MRNet y validación
  externa en Croacia (zero-shot y fine-tuning). Es la fuente de la fila de Swin-Tiny de la tabla
  comparativa y de los resultados de Croacia; sus salidas están en `results_threshold/`.
- `swin_tiny_baseline.ipynb`, `swin_tiny_baseline_alt.ipynb` — dos entrenamientos de una sola semilla
  (en `checkpoints/swin_baseline/` y `checkpoints/swin_tiny_baseline/`), como referencia previa al
  multiseed.

## Nota sobre el lío de nombres heredado

Históricamente, el modelo canónico se entrenó en una carpeta llamada `swin_small`, pero la
arquitectura es Swin-Tiny (profundidad de bloques [2,2,6,2]), no Swin-Small. Aquí ya está todo
renombrado a `swin_tiny`. La variante Swin-Small auténtica (49 M, [2,2,18,2]), que no converge de
forma estable, está en `experiments/swin_small/`. Ver [MODELS.md](../../MODELS.md).
