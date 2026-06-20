# Swin-Small (auténtica)

Esta carpeta contiene el experimento con la variante **Swin-Small auténtica**
(`swin_small_patch4_window7_224`, ~49 M de parámetros, profundidad de bloques [2,2,18,2]), usada para
discutir el efecto del tamaño del modelo.

- `swin_small_real_multiseed.ipynb` — entrenamiento multiseed. Bajo el mismo protocolo que las
  arquitecturas compactas, la mayoría de las semillas colapsan hacia la solución trivial y el modelo
  no converge de forma estable; su AUC de validación queda cerca del azar. Checkpoints en
  `checkpoints/swin_small_REAL_multiseed/`.

Este resultado es justo el que respalda la elección de Swin-Tiny: aumentar la capacidad no mejora el
rendimiento en MRNet. El Swin-Tiny que sí se usa en el trabajo está en
[`experiments/swin_tiny/`](../swin_tiny/).
