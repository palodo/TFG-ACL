# Clasificador de Roturas de LCA (ACL)

Este repositorio contiene el código del Trabajo de Fin de Grado (TFG) para la detección y clasificación automática de lesiones del **Ligamento Cruzado Anterior (LCA)** a partir de volúmenes de Resonancia Magnética (RM) utilizando técnicas de Aprendizaje Profundo.

El sistema procesa los tres planos anatómicos (sagitario, coronal y axial) mediante una arquitectura multi-plano basada en **Swin Transformer** (Swin-Small con agregación multiseed).

---

## 📁 Estructura del Proyecto

* **`src/`**: Módulos compartidos en Python (definición del modelo, carga de datos, transformaciones y entrenamiento).
* **`models/`**: Notebooks de Jupyter para el entrenamiento y ajuste inicial de las arquitecturas (ViT-Tiny, Swin, ResNet50, etc.).
* **`final_model/`**: Notebooks y scripts para la validación del modelo elegido (Swin Small) y su evaluación externa (dataset de Croacia).
* **`scripts/`**: Herramientas útiles de línea de comandos, como `predict_patient.py` para realizar inferencias completas de un paciente a partir de sus archivos DICOM.
* **`app/`**: Aplicación web completa (Frontend en React + Backend en Node.js) que permite subir un caso en ZIP, previsualizar los cortes y obtener el diagnóstico con mapas de calor (saliency/Grad-CAM).

---

## 🚀 Instalación y Uso Rápido

### 1. Requisitos previos
Asegúrate de tener instalado Python 3.9 o superior y las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

### 2. Ejecutar inferencia en terminal
Para clasificar la resonancia magnética de un paciente en formato ZIP (que contenga las series de cortes DICOM) y ver sus mapas de atención:

```bash
python scripts/predict_patient.py --zip /ruta/al/archivo/paciente.zip
```

### 3. Iniciar la aplicación web
Para arrancar la interfaz web interactiva de diagnóstico, simplemente ejecuta:

```bash
python app/start_server.py
```
Esto instalará dependencias, compilará el frontend React y levantará la aplicación web local en `http://localhost:5000/`.

---

**Autor:** Pablo  
**Universidad de Valencia** (UV)
