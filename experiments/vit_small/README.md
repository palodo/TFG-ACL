# ViT-Small

Entrenamiento y evaluación de ViT-Small (`WinKawaks/vit-small-patch16-224`, ~22 M de parámetros,
dimensión de características 384) como representante de los transformers de atención global. Es una de
las tres arquitecturas principales del trabajo.

## Notebooks

- `vit_small_baseline.ipynb` — entrenamiento de una sola semilla por plano, como primera referencia.
- `vit_small_multiseed.ipynb` — entrenamiento con diez semillas por plano; produce los checkpoints
  `best_{plano}_multiseed_final.pth` en `checkpoints/vit_small_multiseed/`.
- `Threshold_Tuning_and_External_Validation.ipynb` — carga los tres planos multiseed, promedia el
  ensamble, fija el umbral clínico sobre validación (recall máximo con precisión >= 0,75) y evalúa en
  test. Es la fuente de la fila de ViT-Small de la tabla comparativa de la memoria.

## Configuración

El pooling entre cortes es *attention* en sagital y axial y *max* en coronal. La clase del modelo es
`ViTSmallMultiSliceClassifier` en `src/models.py`; para cargar los pesos hay que instanciarla con
`pretrained=True`, ya que con `pretrained=False` construye la configuración por defecto de ViT
(dimensión 768) en lugar de la de Small (384).

## Rutas

Los notebooks están en `experiments/vit_small/` y referencian la raíz del repositorio con `../../`
(`../../data`, `../../checkpoints`, `../../src`). Los datos y los pesos no se versionan; ver
[DATA.md](../../DATA.md) y [MODELS.md](../../MODELS.md).
