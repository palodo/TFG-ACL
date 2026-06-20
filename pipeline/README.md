# Pipeline final

Notebooks que integran el sistema completo una vez entrenados los modelos por arquitectura: el
ensamble multi-vista, la calibración del umbral, la validación externa y la inferencia sobre un
estudio real.

## Notebooks

- `Threshold_Tuning_and_External_Validation.ipynb` — carga los tres planos del modelo multiseed,
  promedia el ensamble, busca el umbral clínico sobre validación (recall máximo con precisión ≥ 0,75),
  lo aplica al test de MRNet y evalúa la transferencia a Croacia, incluido el experimento de
  *fine-tuning*. Es la fuente de las métricas de ensamble de la memoria.
- `Swin_Small_Model_Selection_Validation.ipynb` — comparación de los modelos en validación y
  justificación de la elección de arquitecturas compactas.
- `Ensemble_Swin_Attention_Maps.ipynb` — mapas de atención y saliencia del ensamble.
- `Process_Patient_DICOM.ipynb` — recorrido completo desde una serie DICOM real hasta la predicción y
  los mapas de explicabilidad, equivalente a lo que hace la aplicación web.

## Resultados guardados

`results/` y `results_threshold/` contienen las tablas y figuras generadas (CSV de comparación,
curvas ROC, matrices de confusión); `attention_maps_results/` los mapas de explicabilidad de casos
concretos.

Los notebooks usan rutas absolutas a `data/` y `checkpoints/` (ver [DATA.md](../DATA.md) y
[MODELS.md](../MODELS.md)); hay que ajustarlas si el repositorio está en otra ruta.
