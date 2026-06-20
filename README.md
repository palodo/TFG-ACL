# Detección de Lesiones del Ligamento Cruzado Anterior (LCA) mediante Deep Learning

Este proyecto corresponde a un Trabajo de Fin de Grado (TFG) de la Universidad de Valencia enfocado en la clasificación automática de roturas del ligamento cruzado anterior (LCA/ACL) a partir de resonancias magnéticas (RM) de rodilla.

El sistema utiliza un enfoque de diagnóstico de extremo a extremo (*end-to-end*) sin segmentación anatómica manual previa, combinando los tres planos de visión clínicos (sagitario, coronal y axial) mediante un ensamble multi-vista.

---

## Materiales y Conjuntos de Datos

En este estudio se emplean dos conjuntos de datos independientes para validar tanto el rendimiento interno como la capacidad de transferencia clínica real:

1. **Dataset Principal (MRNet - Universidad de Stanford):**
   * **Total:** 1.250 estudios de RM tridimensionales (sagitario, coronal y axial).
   * **Distribución:** 875 casos para entrenamiento, 188 para validación y 187 para prueba final.
   * **Prevalencia:** Aproximadamente el 21% de los casos presentan rotura del LCA, lo que simula el desbalance clínico real.

2. **Dataset de Validación Externa (Hospital Clínico de Rijeka, Croacia):**
   * **Total:** 917 estudios anotados. Para hacerlos compatibles, se unificaron las roturas parciales y completas como casos positivos.
   * **Uso:** Validación *zero-shot* (transferencia directa del modelo sin reentrenamiento) y experimentos de adaptación de dominio (*fine-tuning*) con una partición de prueba local de 138 casos.

---

## Arquitecturas Comparadas

Se evalúa y contrasta el comportamiento de tres enfoques de visión artificial de distinta naturaleza:

* **ResNet50 (Convolucional):** Red clásica basada en circunvoluciones locales y conexiones residuales (23.5 millones de parámetros).
* **ViT-Small (Vision Transformer Global):** Modelo de autoatención global que procesa la imagen dividida en parches (22 millones de parámetros).
* **Swin-Tiny (Vision Transformer Jerárquico):** Evolución de ViT que restringe la atención a ventanas locales desplazadas, permitiendo capturar características a múltiples escalas de forma eficiente (27.8 millones de parámetros).

*Nota metodológica:* Todos los modelos integran un módulo automático de selección de cortes relevantes (MobileNetV2) que filtra y extrae las imágenes donde el ligamento es visible (5 cortes sagitales, 10 coronales y 10 axiales), seguidos de un mecanismo de agregación por atención (*Attention Pooling*).

---

## Resultados Clave

### 1. Rendimiento Interno (Prueba en MRNet)
Los tres modelos logran un excelente desempeño en la clasificación de lesiones sobre el conjunto de test propio de Stanford:
* **ResNet50:** Obtiene el mayor rendimiento discriminativo global con un **AUC de 0.9545** (Sensibilidad: 88.37%, F1-Score: 0.7379).
* **Swin-Tiny:** Destaca por su alta seguridad diagnóstica, alcanzando la mayor sensibilidad (**Recall del 90.70%**, con solo 4 falsos negativos en el test). Su AUC es de **0.9536**.
* **ViT-Small:** Ofrece el mejor balance general con un F1-Score de **0.7525** (AUC: 0.9433).

### 2. Capacidad de Generalización (Validación en Croacia)
Al evaluar los modelos entrenados en Stanford directamente sobre el conjunto de Croacia sin realizar ajustes (*zero-shot*), se observa una caída de rendimiento debido al cambio de escáner y población (cambio de dominio):
* **Transformers (Swin-Tiny y ViT-Small):** Demuestran ser más robustos. **Swin-Tiny** obtiene el mejor rendimiento general con un **AUC de 0.8821**, mientras que **ViT-Small** mantiene la sensibilidad más alta (**68.28%**).
* **CNN (ResNet50):** Sufre el mayor impacto clínico, con su **AUC cayendo hasta 0.7847**, lo que evidencia un mayor sobreajuste a las particularidades visuales de las imágenes originales de Stanford.

### 3. Adaptación de Dominio (*Fine-Tuning*)
Para solucionar el cambio de dominio, se realizó un reentrenamiento parcial (*fine-tuning*) de **Swin-Tiny** usando casos locales de Croacia. Como resultado, la capacidad discriminativa en este conjunto externo aumentó de un **AUC de 0.9236 a 0.9362** (+0.0126), demostrando la viabilidad de adaptar el sistema a nuevos centros hospitalarios.

---

## Estructura del Código

* **`src/`**: Código fuente de carga de datos, definición de las arquitecturas y funciones de entrenamiento.
* **`scripts/predict_patient.py`**: Script unificado de diagnóstico. Permite clasificar una resonancia magnética en ZIP y obtener mapas de Grad-CAM.
* **`app/`**: Aplicación web interactiva (React frontend + Node backend) que despliega de manera gráfica el asistente de diagnóstico.
* **`final_model/`**: Jupyter Notebooks enfocados en el análisis final y la validación cruzada.

---

## Inicio Rápido

1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Realizar un diagnóstico desde terminal:
   ```bash
   python scripts/predict_patient.py --zip /ruta/al/caso_paciente.zip
   ```
3. Ejecutar la interfaz web:
   ```bash
   python app/start_server.py
   ```

---

**Autor:** Pablo
**Trabajo de Fin de Grado** - Universidad de Valencia (UV)
