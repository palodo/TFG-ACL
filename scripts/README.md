# Scripts

Utilidades en Python, pensadas para ejecutarse desde la raíz del repositorio.

- `predict_patient.py` — diagnóstico de un estudio empaquetado en ZIP, con mapas de explicabilidad.
- `recompute_vit_small_val.py` — recalcula el ensamble de ViT-Small en validación y test de MRNet con
  el umbral clínico, a partir de los checkpoints multiseed. Reproduce la fila de ViT-Small de la tabla
  comparativa de la memoria.
- `generate_croacia_cm.py` — matrices de confusión de Croacia *zero-shot* en el formato de la memoria.
- `generate_fn_gradcam.py` — figura de los falsos negativos comunes con la región de mayor saliencia.
- `generate_tfg_diagrams.py`, `generate_tfg_heatmaps.py`, `generate_tfg_heatmaps_allmodels.py`,
  `gen_pipeline_v3.py` — figuras del documento.
- `convert_croatia_to_npy.py`, `generate_croatia_slice_indices.py`, `verify_croatia_npy.py` —
  preparación de los volúmenes y los índices de cortes de Croacia.

Los scripts que cargan modelos necesitan los pesos descritos en [MODELS.md](../MODELS.md) y un
entorno con PyTorch. Varios usan rutas absolutas a `data/` y `checkpoints/`.
