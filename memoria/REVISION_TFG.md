# Revisión técnica del TFG y guía de redacción por capítulos

**Proyecto:** Diagnóstico de lesiones de LCA en RM de rodilla
**Documento revisado:** `pdf/tfg.pdf` (52 páginas)
**Fecha de revisión:** 2026-06-11

Este documento está pensado para que lo uses como guion al reescribir el TFG.
Cada sección indica: (1) qué dice ahora el PDF, (2) qué dice realmente el código/notebooks,
(3) qué hay que corregir o añadir. Los puntos marcados con ** CRÍTICO** son errores
conceptuales o factuales que un tribunal puede detectar y que conviene arreglar sí o sí.

---

## 0. Resumen ejecutivo de hallazgos

| # | Severidad | Hallazgo | Dónde |
|---|-----------|----------|-------|
| 1 |  CRÍTICO | El "Swin-Small" es en realidad **Swin-Tiny** (27.8 M parámetros, no 49.6/49.7 M) | Cap. 3.6, 4.0.4 |
| 2 |  CRÍTICO | Se afirma resolución de entrada **256×256 para Swin**; el código redimensiona **siempre a 224×224** | Cap. 2.2, 3.6 |
| 3 |  CRÍTICO | El selector de cortes es **MobileNetV2**, no MobileNetV3 | Cap. 3.2 |
| 4 |  IMPORTANTE | El CSV guardado de Croacia da **AUC 0.5058**, pero el notebook y el PDF reportan **0.8821**: hay una ejecución incoherente sin resolver | `final_model/results_threshold/comparacion_metricas.csv` |
| 5 |  IMPORTANTE | Tablas de regularización con discrepancias puntuales (early stopping coronal de ViT, etc.) | Cap. 3.4–3.6 |
| 6 |  MENOR | Capítulo 5 (Conclusiones y Trabajo futuro) está **vacío** | Cap. 5 |
| 7 |  MENOR | Métricas de F1/precisión en validación se calculan con umbral fijo 0.5 dentro del entrenamiento, pero el ensamble usa umbral calibrado: aclarar para no confundir | `src/training_utils.py`, Cap. 4 |
| 8 |  MENOR | La app rellena por defecto un **nombre de paciente real** ("Marc Alberola Guillot") cuando faltan metadatos DICOM | `scripts/predict_patient.py` (corregido) |

---

## Cómo está construido el sistema (mapa real del código)

Antes de los capítulos, este es el flujo real verificado en el código, porque varias cosas
del PDF no coinciden con él:

- **Datos:** volúmenes MRNet en `.npy` por plano (`data/{train,val,test}/{sagittal,coronal,axial}/XXXX.npy`).
  Splits: train 875 / val 188 / test 187 (coincide con el PDF).
- **Selección de cortes:** `src/models.py::SimpleCNNSelector` con backbone **`mobilenetv2_100`** (timm).
  Dos selectores entrenados (sagital y coronal). El axial usa cortes centrales consecutivos.
  Índices precalculados y cacheados en `data/slice_indices_final/*.json` (clase `OptimizedMRNetDataset`).
- **Clasificadores multi-corte** (`src/models.py`):
  - `CNNMultiSliceClassifier` → ResNet50 (`torchvision.models.resnet50`), feature_dim 2048.
  - `ViTMultiSliceClassifier` → `google/vit-base-patch16-224` (HuggingFace), feature_dim 768.
  - `SwinMultiSliceClassifier` → **`swin_tiny_patch4_window7_224.ms_in1k`** (timm), feature_dim 768. *(Aquí está el error de nombre.)*
  - Todos comparten la misma cabeza: pooling por cortes (attention / max / mean) + clasificador
    `Dropout → Linear(feat,256) → ReLU → Dropout → Linear(256,1)`, salida de 1 logit (BCE).
- **Pooling por plano:** la `AttentionPooling` aprende un peso por corte (softmax sobre K cortes).
- **Entrenamiento** (`src/training_utils.py`): AdamW, `SmoothedBCELoss` con `pos_weight` por desbalance,
  label smoothing, gradient clipping opcional, early stopping por **AUC** de validación,
  ReduceLROnPlateau o WarmupCosine opcional. Multi-semilla (10 semillas, 42–51).
- **Ensamble:** promedio simple de las 3 probabilidades de plano. Umbral calibrado en validación
  maximizando recall con precisión ≥ 0.75.
- **App:** frontend React (`app/frontend`) + backend Node/Express (`app/server.js`) que invoca
  `scripts/predict_patient.py` (extrae el ZIP DICOM, selecciona cortes, ejecuta el ensamble y
  genera mapas de saliencia/atención).

---

## Capítulo 1 — Introducción y Estado del Arte

**Estado actual:** correcto en estructura (contexto, objetivos, MRNet, ResNet, ViT, Swin, comparativa).

**Qué revisar / añadir:**
- En 1.3 mencionas que tu selector de cortes es una contribución metodológica propia. Bien, pero
  **asegúrate de describirlo con el backbone correcto (MobileNetV2)** para que coincida con métodos.
- Convendría anticipar en objetivos que la comparación es **a igualdad de pipeline** (mismo selector,
  misma cabeza de atención, mismo protocolo multi-semilla), porque es tu principal fortaleza
  metodológica y ahora no se subraya.
- No hay error conceptual aquí. Es el capítulo más sólido.

---

## Capítulo 2 — Materiales

**Estado actual:** describe MRNet, el conjunto de Croacia, y el preprocesamiento.

### CRÍTICO #2 — Resolución de entrada
- El PDF (2.2 y 3.6) dice: *"todas las imágenes fueron redimensionadas a 224×224"* y luego que
  *"Swin-Small se configuró utilizando imágenes de entrada de mayor resolución espacial (256×256)"*.
  **Son afirmaciones contradictorias y la segunda es falsa.**
- En el código, **todas** las transformaciones (`src/data_loader.py::get_train_transform` y
  `get_val_test_transform`) hacen `transforms.Resize((224,224))`. La variable `IMAGE_SIZE = 256`
  del notebook de Swin **no se usa** en el pipeline de datos.
- **Acción:** elimina la frase del 256×256 para Swin. Todos los modelos usan 224×224. Si quieres
  conservar la idea de "mayor resolución", tendrías que reentrenar con un transform a 256, cosa
  que no se hizo.

**Otros puntos:**
- La normalización Min-Max y la réplica a 3 canales están bien descritas y coinciden con el código.
- Las cifras de splits MRNet (875/188/187) y de Croacia (917; 690 sin rotura / 227 con rotura;
  fine-tuning 641/138/138) coinciden con CSVs y notebooks.
- Sugerencia: indica explícitamente que MRNet aquí es un **re-split a nivel de paciente** propio
  (total 1250), distinto del split oficial de Stanford (1130/120), para que no confundan tus cifras
  con las del paper original.

---

## Capítulo 3 — Métodos

### 3.2 Selección automática de cortes

### CRÍTICO #3 — Backbone del selector
- El PDF dice **MobileNetV3** ("se desarrolló un módulo propio de selección automática de cortes
  basado en una arquitectura MobileNetV3").
- El código usa **`mobilenetv2_100`** (timm) tanto en `src/models.py::ACLSliceClassifier` /
  `SimpleCNNSelector` como en `scripts/predict_patient.py`. El propio frontend lo muestra como
  "Localizador (MobileNetV2)".
- **Acción:** cambiar todas las menciones a **MobileNetV2**.

**Bien:** la descripción de "puntuar cada corte → ordenar → top-K" es exacta. K=5 sagital,
K=10 coronal (CNN-based), K=10 axial (cortes centrales). Coincide con el código.

### 3.4 ResNet50
- Familia, conexiones residuales, 23.5 M parámetros: correcto (ResNet50 ≈ 25.5 M en total, pero
  como usas el backbone sin la fc final, ~23.5 M es defendible; puedes precisar "≈ 23.5 M en el
  backbone de extracción de características").
- Pooling por plano (sagital attention, coronal max, axial attention): **coincide** con
  `PLANE_CONFIG` del notebook `cnn_multiseed`.
- **Revisar Tabla 3.2 (regularización):** los valores de dropout (0.42/0.47/0.42 y 0.32/0.37/0.32)
  y label smoothing (0.17/0.22/0.17) coinciden con el notebook. El early stopping (12/10/12) hay
  que **verificarlo** contra `PLANE_SPECIFIC_CONFIG` de `cnn_multiseed` (el sagital es 12; confirma
  coronal y axial célula a célula).

### 3.5 ViT-Base
- `google/vit-base-patch16-224`, 12 bloques, embedding 768, 12 cabezas, ~86 M parámetros: **correcto**
  y coincide con el código y el checkpoint (`position_embeddings (1,197,768)`).
- Pooling (sagital attention, coronal max, axial attention): coincide con `cnn`/`vit` `PLANE_CONFIG`.
- **Revisar Tabla 3.3:** el PDF pone early stopping coronal = **15 épocas**, pero el notebook
  `vit_multiseed` tiene coronal = **25 épocas** (sagital 25, coronal 25, axial 15). Corrige el coronal.
- Label smoothing 0.05/0.05/0.10: coincide.

### 3.6 Swin

### CRÍTICO #1 — Es Swin-Tiny, no Swin-Small
- El PDF lo llama **Swin-Small** y le asigna **49.6 M** (Tabla 3.1) / **49.7 M** (texto 3.6.1).
- El código instancia **`swin_tiny_patch4_window7_224.ms_in1k`** en `SwinMultiSliceClassifier`
  (`src/models.py`, feature_dim **768**). Existe una clase `SwinBaseMultiSliceClassifier`
  (1024-dim) pero **no es la que se usa** en el modelo final ni en la app.
- Verificación directa sobre el checkpoint final
  (`checkpoints/swin_tiny_multiseed/best_sagittal_multiseed_final.pth`):
  - `embed_dim = 96`, profundidades por etapa **[2, 2, 6, 2]**, **27.8 M** parámetros.
  - Swin-**Tiny** = depths [2,2,6,2], embed 96, ≈28 M. Swin-**Small** = depths [2,2,18,2], ≈50 M.
  - Es inequívocamente **Swin-Tiny**.
- **Acción (elige una vía):**
  - **(A) Honesta y rápida:** renombrar a **Swin-Tiny** en todo el documento (título 3.6.1, Tabla 3.1,
    figuras, capítulo 4). Cambiar "49.6/49.7 M" por "≈28 M". Esto **no perjudica** tu narrativa: de
    hecho la refuerza, porque un Swin-Tiny (28 M) iguala a un ViT-Base (86 M) — es un argumento de
    eficiencia muy potente.
  - **(B) Solo si reentrenas:** cambiar el backbone a `swin_small_patch4_window7_224` y volver a
    entrenar las 10 semillas × 3 planos. Mucho trabajo y riesgo; no recomendado a estas alturas.
- También corrige la frase de "256×256 para Swin" (ver Crítico #2).
- El feature_dim correcto de Swin-Tiny es 768 (no 1024); si en algún punto mencionas 1024, bórralo.

**Bien en 3.6:** la descripción conceptual del shifted-window attention, el sesgo de posición
relativa B, el Patch Merging jerárquico y el pooling por atención en los 3 planos son correctos y
coinciden con `PLANE_CONFIG` de `swin_tiny_multiseed` (los 3 planos usan attention).

**Detalle de entrenamiento Swin que sí coincide:** AdamW, lr inicial 1.3e-4, ReduceLROnPlateau
factor 0.5, grad clipping norma 1.0, dropout axial input 0.0. Tabla 3.4 (0.42/0.42/0.00 input,
0.32/0.32/0.20 dense, ls 0.17/0.17/0.05, early stopping 12/40/30) **coincide** con el notebook.

---

## Capítulo 4 — Resultados y Discusión

### 4.0.1 Diseño experimental
- El protocolo (10 semillas 42–51, selección de checkpoint por validación, test solo como
  estimación final, umbral τ* = argmax recall s.a. precisión ≥ 0.75, ensamble por promedio simple)
  es correcto y coincide con `find_optimal_threshold_recall` y los notebooks de threshold.
- **Punto a aclarar (#7):** dentro del entrenamiento (`evaluate` en `training_utils.py`) el F1 se
  calcula con `np.round` (umbral 0.5). Esos F1 "internos" **no** son los del ensamble calibrado.
  En el capítulo 4 las métricas operativas vienen del notebook de threshold (umbral calibrado), que
  es lo correcto. Solo asegúrate de no mezclar ambas fuentes de F1 en el texto.

### 4.0.2–4.0.4 Resultados por arquitectura
- Tablas de estabilidad multi-semilla (ResNet50, ViT-Base, Swin): los AUC medios/desv./mín/máx que
  aparecen en el PDF coinciden con los outputs de los notebooks.  (p.ej. Swin axial 0.9560±0.0123,
  sagital 0.9223±0.0787, coronal 0.7748±0.1348 — esa alta varianza del coronal está bien comentada).
- Umbrales por arquitectura: ResNet50 τ=0.5445, ViT-Base τ=0.5135, Swin τ=0.3734 — coinciden con los
  notebooks y con `scripts/predict_patient.py`.
- Métricas de test del ensamble Swin (AUC 0.9536, recall 0.9070, 4 FN, 23 FP) coinciden con el
  output ejecutado (`final_model/Threshold_Tuning_and_External_Validation.ipynb`).
- **Recuerda:** al pasar de "Swin-Small" a "Swin-Tiny" hay que sustituir el nombre también en todas
  las tablas 4.9–4.13, figuras 4.5–4.8 y en la discusión.

### 4.0.6 Validación externa (Croacia) —  IMPORTANTE #4
- El PDF reporta **AUC zero-shot de Swin en Croacia = 0.8821** (Tabla 4.14), que coincide con el
  **output ejecutado** del notebook (`AUC: 0.8821`, recall 0.5683, esp. 0.9638).
- **Pero** el fichero guardado `final_model/results_threshold/comparacion_metricas.csv` dice
  **Croacia AUC = 0.5058**, recall 0.7533, especificidad 0.2406. Es decir, hay **dos resultados
  incompatibles** para el mismo experimento.
- Causa probable: el CSV es de una ejecución antigua (índices de cortes mal generados, o volúmenes
  `croatia_npy_volumes_final` vs `croatia_npy_volumes_full`, o sin `CROATIA_REGENERATE_INDICES`).
  El notebook actual regenera índices y da 0.8821.
- **Acción antes de entregar:** re-ejecuta el notebook de cabo a rabo y confirma que el número que
  pones (0.8821) es reproducible y que el CSV viejo se sobrescribe. Si no, un revisor que abra el CSV
  verá 0.5058 y será un problema de credibilidad. Borra o regenera el CSV obsoleto.
- Nota metodológica: la validación externa usa **solo el plano sagital** (no el ensamble de 3),
  porque Croacia solo tiene sagital fiable. Déjalo dicho explícitamente (ya lo está, bien).

### 4.0.7 Fine-tuning en Croacia
- AUC test Croacia antes 0.9236 → después 0.9362 (+0.0126): coincide con el output ejecutado.
  (Ojo: el "antes" zero-shot sobre el *subconjunto de test* de Croacia es 0.9236, distinto del
  0.8821 sobre el *conjunto completo*; conviene aclarar que son poblaciones distintas para que no
  parezca contradicción.)
- Congelar backbone 2 épocas + ajuste completo, lr 2e-4, pos_weight 3.0314, dropout 0.3/0.2:
  coincide con la llamada a `train_model`.

### 4.0.8 Discusión
- Bien planteada. Sugerencia: cuando renombres a Swin-Tiny, refuerza la conclusión de **eficiencia**
  (28 M parámetros compitiendo con ViT-Base de 86 M). Es tu mejor argumento y ahora se pierde.

---

## Capítulo 5 — Conclusiones y Trabajo futuro   #6

**Está vacío en el PDF** (solo aparecen los títulos 5.1 y 5.2). Hay que redactarlo. Guion sugerido:

**5.1 Conclusiones**
1. Se ha construido un pipeline end-to-end (selector MobileNetV2 + clasificador multi-corte con
   atención + ensamble multi-vista) que detecta rotura de LCA con AUC ~0.95 en test interno.
2. La comparación a igualdad de pipeline muestra que **no hay un ganador absoluto**: ResNet50 logra
   el mayor AUC en test (0.9545), ViT-Base la mejor especificidad/precisión, y Swin-Tiny la mayor
   sensibilidad (0.9070, 4 FN), que es la métrica clínicamente prioritaria.
3. El análisis multi-semilla es imprescindible en datasets médicos moderados: la varianza entre
   inicializaciones (sobre todo en el plano coronal) puede cambiar las conclusiones.
4. Existe un **domain shift** claro hacia Croacia: la sensibilidad cae aunque el AUC se mantenga
   razonable; el fine-tuning lo mitiga parcialmente (+0.013 AUC).
5. Un Transformer jerárquico de solo 28 M parámetros (Swin-Tiny) iguala a un ViT-Base de 86 M:
   argumento de eficiencia para despliegue clínico.

**5.2 Trabajo futuro**
- Calibración de probabilidades multi-centro (Platt/temperature scaling) y umbrales por dominio.
- Selector de cortes también para el plano axial (hoy son cortes centrales fijos).
- Validación externa más amplia y prospectiva; estudiar variabilidad de protocolo/escáner.
- Modelos 3D o atención inter-plano aprendida (hoy el ensamble es promedio sin parámetros).
- Explicabilidad clínica validada por radiólogos (los mapas de saliencia/atención de la app son un
  primer paso, pero no están validados cuantitativamente).

---

## Anexo A — Revisión de los notebooks (¿hay errores de entrenamiento?)

**No hay errores que invaliden los resultados.** El pipeline de entrenamiento es metodológicamente
correcto: split a nivel de paciente, augmentation solo en train, normalización consistente,
pos_weight para desbalance, early stopping por AUC de validación, test intocado para decisiones.
Observaciones menores (mejoras, no errores):

1. **F1 con umbral 0.5 durante el entrenamiento** (`evaluate`): es solo informativo; las métricas
   clínicas definitivas usan el umbral calibrado. No afecta a resultados, pero conviene no citarlo
   como F1 "final".
2. **`WarmupCosineScheduler` usa `optimizer.defaults['lr']`** como lr base. Funciona, pero si se
   reanudara un entrenamiento o se cambiara el lr por grupos, podría dar sorpresas. No afecta a tus
   ejecuciones actuales.
3. **`init_attention_weights` reinicializa la pooling con Xavier** tras cargar pesos preentrenados:
   correcto (la cabeza de atención no viene de ImageNet). Bien.
4. **El selector se ejecuta sobre volúmenes normalizados Min-Max global** y luego ImageNet-normalize:
   consistente con `OptimizedMRNetDataset`. Bien.
5. **Croacia (Anexo importante):** la incoherencia 0.5058 vs 0.8821 (ver #4) es lo único que hay que
   resolver reproduciendo el notebook completo. Es un problema de *artefacto guardado*, no de método.
6. **Reproducibilidad:** se fija semilla, `cudnn.deterministic=True` y `benchmark=False`. Bien. Aun
   así, DataParallel + atención puede tener pequeñas no-determinismos; menciónalo si reportas
   cifras al cuarto decimal.

---

## Anexo B — Revisión de la app

La "app.py" como tal no existe. La aplicación son tres piezas:
`app/server.js` (Node/Express, BFF), `scripts/predict_patient.py` (motor de inferencia) y
`app/frontend` (React). Revisión y estado:

**Corregido en esta revisión (`scripts/predict_patient.py`):**
- **Privacidad #8:** los valores por defecto de los metadatos del paciente eran un **nombre real**
  ("Marc Alberola Guillot", ID "RM-RODILLA-IZQ", edad "28Y", fecha concreta). Se han sustituido por
  valores anónimos genéricos ("Paciente Anónimo", "ANON-ID", etc.). Importante para una demo pública.

**Recomendado (no aplicado para no arriesgar el funcionamiento; ver detalle en el código):**
- `server.js`: los ZIP subidos se conservan indefinidamente en `app/uploads` ("retained for
  presentation"). Para una demo está bien, pero implica retención de datos clínicos. Añadir limpieza
  por antigüedad o un endpoint de borrado si se publica.
- `server.js`: los endpoints `/api/analyze` y `/api/analyze-existing` están casi duplicados; podrían
  unificarse en una función `runInference(zipPath, model, res)`.
- `server.js`: no hay límite de concurrencia de procesos Python; cada petición lanza un proceso que
  carga modelos en GPU/CPU. Para demo monollamada está bien; para uso real, encolar.
- `predict_patient.py`: el plano axial no tiene selector aprendido (usa `selector_prob = 0.5` de
  relleno); es coherente con el método del TFG, solo documéntalo.

**Sin errores funcionales bloqueantes:** la extracción de ZIP es segura (protección Zip-Slip), la
selección de cortes, la carga de checkpoints (limpieza de prefijo `module.`), el ensamble y el
umbral por modelo (swin 0.3734 / vit 0.5135 / cnn 0.5455) son correctos y coinciden con los
notebooks.
