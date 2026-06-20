# Experimentos por arquitectura

Notebooks de entrenamiento y evaluación de cada arquitectura, una carpeta por familia. Cada notebook
asume que se ejecuta desde su carpeta y referencia los datos y los pesos en la raíz del repositorio
(`../../data`, `../../checkpoints`), que no están versionados (ver [DATA.md](../DATA.md) y
[MODELS.md](../MODELS.md)).

Dentro de cada familia suele haber dos notebooks:

- `*_baseline.ipynb`: entrenamiento de una sola semilla, para una primera referencia.
- `*_multiseed.ipynb`: entrenamiento con diez semillas por plano, que es lo que se usa en el ensamble
  final de la memoria. Guarda en `checkpoints/<arquitectura>/best_{plano}_multiseed_final.pth` el
  modelo seleccionado por AUC de validación de cada plano.

## Carpetas

Las tres arquitecturas que se comparan en la memoria:

- `cnn/` — ResNet50 (convolucional, ~23,5 M).
- `vit_small/` — ViT-Small (atención global, ~22 M).
- `swin_tiny/` — Swin-Tiny (atención jerárquica, ~28 M).

El resto son experimentos de apoyo para discutir el efecto del tamaño del modelo y comparar con otras
familias:

- `vit/` — ViT-Base (86 M) y sus utilidades de atención.
- `vit_tiny/` — ViT-Tiny.
- `swin_base/` — Swin-Base.
- `swin_small/` — incluye la variante Swin-Small auténtica (49 M), que no converge de forma estable
  bajo el mismo protocolo (`swin_SMALL_REAL_multiseed.ipynb`).
- `convnext/` — ConvNeXt, como referencia convolucional adicional.

## Estructura común de un notebook

La configuración por plano (pooling, regularización) está al principio. El pooling entre cortes es
*attention* en sagital y axial y *max* en coronal, igual en todas las arquitecturas. Después se
entrena cada plano por separado, se cargan los tres modelos, se promedia el ensamble y se calibra el
umbral de decisión sobre validación (recall máximo con precisión ≥ 0,75) antes de evaluar en test.
