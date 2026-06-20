# Datos

Los conjuntos de datos no se incluyen en el repositorio (`data/` está excluido en `.gitignore`).
Ninguno de los dos es redistribuible directamente; hay que obtenerlos de su fuente original.

## MRNet (Stanford)

Dominio principal. Se descarga desde el sitio oficial de Stanford, que exige registro y aceptar las
condiciones de uso: https://stanfordmlgroup.github.io/competitions/mrnet/

Referencia: Bien et al., *Deep-learning-assisted diagnosis for knee magnetic resonance imaging*,
PLOS Medicine, 2018.

## Validación externa (Hospital Clínico de Rijeka, Croacia)

Conjunto público de RM de rodilla con anotación de LCA en tres grados. En este trabajo las roturas
parciales y completas se unifican como clase positiva.

Referencia: Štajduhar et al., *Semi-automated detection of anterior cruciate ligament injury from
MRI*, 2017.

## Rutas esperadas

El código asume esta organización dentro de `data/`:

```
data/
├── train/{sagittal,coronal,axial}/<caso>.npy     # volúmenes MRNet por plano
├── val/  {sagittal,coronal,axial}/<caso>.npy
├── test/ {sagittal,coronal,axial}/<caso>.npy
├── train-acl.csv , val-acl.csv , test-acl.csv    # etiquetas binarias de rotura de LCA
├── slice_indices_final/                          # índices de cortes seleccionados (precalculados)
│   ├── {train,val,test}_{sagittal,coronal,axial}_indices.json
│   └── croatia_sagittal_indices.json
└── croatia_npy_volumes_full/                      # volúmenes de Croacia + metadata_final.csv
```

Cada volumen es un array de NumPy con forma `(S, H, W)` (cortes, alto, ancho). Los índices de cortes
de `slice_indices_final/` los genera el selector descrito en [MODELS.md](MODELS.md); el notebook
`data_prep/Generate_Croatia_Slice_Indices.ipynb` reproduce los de Croacia.

## Preprocesamiento

Las intensidades se normalizan por estudio (Min-Max), los cortes se redimensionan a 224×224 y se
replican a tres canales para usar pesos preentrenados en ImageNet. El detalle está en la memoria
(capítulo de materiales y métodos) y en `src/preprocessing.py`.
