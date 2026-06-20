# Integración y demostración

Notebooks que no pertenecen a una sola arquitectura, sino que integran el sistema o lo demuestran.
La evaluación de cada modelo (entrenamiento, ensamble y umbral) está en `experiments/<modelo>/`;
aquí queda lo transversal.

## Notebooks

- `Comparacion_de_Modelos_Validacion.ipynb` — compara los modelos en validación y justifica la
  elección de arquitecturas compactas. Genera `results/model_comparison_validation.csv`.
- `Inferencia_desde_DICOM.ipynb` — toma una serie DICOM real, reconstruye los volúmenes, selecciona
  cortes, ejecuta el ensamble y produce la predicción y los mapas de explicabilidad. Es el mismo
  recorrido que hace la aplicación web. Deja los datos intermedios en `patient_volume/`.
- `Mapas_de_Explicabilidad.ipynb` — genera los mapas de atención y saliencia del ensamble.
  Resultados en `attention_maps_results/`.

Los notebooks usan rutas absolutas a `data/` y `checkpoints/` (ver [DATA.md](../DATA.md) y
[MODELS.md](../MODELS.md)); hay que ajustarlas si el repositorio está en otra ruta.
