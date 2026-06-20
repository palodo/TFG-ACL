# Swin-Tiny

Entrenamiento y evaluación de Swin-Tiny (`swin_tiny_patch4_window7_224`, ~28 M de parámetros), la
arquitectura de atención jerárquica del trabajo y una de las tres principales.

## Notebooks

- `swin_tiny_baseline.ipynb` — entrenamiento de una sola semilla por plano.
- `swin_tiny_multiseed.ipynb` — entrenamiento con diez semillas por plano.
- `Threshold_Tuning_and_External_Validation.ipynb` — ensamble de los tres planos, umbral clínico
  sobre validación (recall máximo con precisión >= 0,75), test de MRNet y validación externa en
  Croacia (zero-shot y fine-tuning). Es la fuente de la fila de Swin-Tiny de la tabla comparativa y
  de los resultados de Croacia de la memoria; sus salidas están en `results_threshold/`.

A diferencia de las otras arquitecturas, Swin-Tiny usa *attention pooling* en los tres planos.

## Nota sobre los checkpoints

El notebook de umbral carga `checkpoints/swin_small_multiseed/`. El nombre de esa carpeta es
heredado y confuso: corresponde al modelo que la memoria llama Swin-Tiny, no a la variante Swin-Small
auténtica (49 M), que está en `checkpoints/swin_small_REAL_multiseed/` y no converge de forma
estable. Ver [MODELS.md](../../MODELS.md).
