# Detección de roturas del ligamento cruzado anterior en RM de rodilla

Código del Trabajo de Fin de Grado (Grado en Ciencia de Datos, Universitat de València) sobre
clasificación automática de roturas del ligamento cruzado anterior (LCA) a partir de resonancias
magnéticas de rodilla.

El sistema no segmenta el ligamento ni depende de anotaciones manuales por corte. A partir del
volumen completo, un selector aprendido elige los cortes más informativos de cada plano, un
clasificador procesa esos cortes y la decisión final se obtiene combinando los tres planos
anatómicos (sagital, coronal y axial). Sobre las probabilidades del conjunto de validación se fija
un umbral de decisión con criterio clínico (maximizar sensibilidad manteniendo una precisión
mínima del 75 %).

La pregunta de fondo del trabajo no es solo "¿funciona?", sino si hace falta un modelo grande para
que funcione. La conclusión es que no: las variantes compactas (ViT-Small de 22 M y Swin-Tiny de
28 M) igualan o superan a sus versiones grandes en este conjunto de datos.

## Datos

Se usan dos conjuntos, ninguno incluido en el repositorio por tamaño y licencia (ver
[DATA.md](DATA.md) para obtenerlos y colocarlos en las rutas esperadas):

- **MRNet** (Stanford), dominio principal: 1.250 estudios con los tres planos. Particiones a nivel
  de paciente: 875 entrenamiento, 188 validación, 187 test. La prevalencia de rotura ronda el 21 %,
  coherente con el desbalance clínico real.
- **Hospital Clínico de Rijeka, Croacia**, validación externa: 917 estudios. Las roturas parciales y
  completas se unifican como clase positiva. Se emplea para transferencia directa (*zero-shot*) y
  para un experimento de adaptación de dominio.

## Enfoque

1. Selección de cortes por plano con un MobileNetV2 entrenado para puntuar relevancia anatómica
   (5 cortes en sagital, 10 en coronal; en axial se usan cortes centrales).
2. Backbone por plano sobre los cortes seleccionados, con agregación entre cortes mediante
   *attention pooling* en sagital y axial y *max pooling* en coronal.
3. Ensamble multi-vista: media de las probabilidades de los tres planos.
4. Umbral de decisión calibrado en validación (recall máximo con precisión ≥ 0,75).

Se comparan tres arquitecturas bajo el mismo protocolo (mismo selector, mismo pooling por plano,
diez semillas por plano):

- ResNet50, convolucional (~23,5 M parámetros).
- ViT-Small, atención global por parches (~22 M).
- Swin-Tiny, atención jerárquica por ventanas desplazadas (~28 M).

## Resultados (ensamble multi-vista, MRNet)

| Modelo    | AUC val | AUC test | Recall test | F1 test |
|-----------|:------:|:-------:|:----------:|:------:|
| ResNet50  | 0,9836 | 0,9545  | 0,8837     | 0,7379 |
| ViT-Small | 0,9837 | 0,9433  | 0,8837     | 0,7525 |
| Swin-Tiny | 0,9837 | 0,9536  | 0,9070     | 0,7459 |

No hay un ganador único: ResNet50 da el mayor AUC en test, Swin-Tiny la mayor sensibilidad (39 de
43 roturas detectadas, 4 falsos negativos) y ViT-Small el mejor F1. En la validación externa sobre
Croacia *zero-shot* el rendimiento baja por el cambio de dominio (Swin-Tiny obtiene el mejor AUC
externo, 0,8821); un *fine-tuning* de Swin-Tiny sobre Croacia sube el AUC de 0,9236 a 0,9362 en ese
dominio. El detalle está en la memoria.

## Organización del repositorio

- `src/` — código compartido: arquitecturas (`models.py`), carga de datos, utilidades de
  entrenamiento.
- `experiments/` — notebooks de entrenamiento por arquitectura (cnn, vit, vit_small, vit_tiny,
  swin_tiny, swin_small, swin_base, convnext), incluyendo los multiseed.
- `pipeline/` — pipeline final integrado: ensamble y calibración de umbral, inferencia desde DICOM,
  mapas de atención.
- `data_prep/` — preparación de los datos externos de Croacia.
- `analysis/` — notebooks de análisis transversal (pipeline optimizado, falsos negativos,
  predicción desde DICOM).
- `scripts/` — utilidades reproducibles (predicción de un paciente, generación de figuras de la
  memoria, recálculo de métricas).
- `app/` — aplicación web del demostrador (frontend en React + servidor Node + motor de inferencia).
- `memoria/` — documento del TFG en LaTeX y su PDF compilado.
- `docs/` — documentación adicional y figuras del pipeline.

Los pesos entrenados (`checkpoints/`) y los datos (`data/`) no se versionan; ver [MODELS.md](MODELS.md)
y [DATA.md](DATA.md).

## Uso

Instalación de dependencias de Python:

```bash
pip install -r requirements.txt
```

Diagnóstico de un estudio (ZIP con la serie DICOM) desde terminal, con mapas de explicabilidad:

```bash
python scripts/predict_patient.py --zip ruta/al/estudio.zip
```

Aplicación web (compila el frontend y arranca el servidor en `http://localhost:5000`):

```bash
python app/start_server.py
```

La inferencia requiere los pesos descritos en [MODELS.md](MODELS.md).

## Memoria

El documento completo está en `memoria/tfgs/` (LaTeX) y se compila con `tectonic`:

```bash
cd memoria/tfgs && tectonic -X compile ejemplo-memoria.tex
```

## Autoría

Pablo López Domínguez. Trabajo de Fin de Grado, Universitat de València. Tutores: José David Martín
Guerrero y Yolanda Vives Gilabert.
