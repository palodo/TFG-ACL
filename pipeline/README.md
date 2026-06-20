# Final Model: Swin Small Model Selection & Validation

## 📁 Estructura

```
final_model/
├── Swin_Small_Model_Selection_Validation.ipynb  # Notebook principal
├── results/                                       # Directorio para resultados
│   ├── model_comparison_validation.csv
│   ├── threshold_tuning_validation.png
│   ├── test_results.png
│   └── external_validation_croatia.png
└── README.md                                      # Este archivo
```

## 📋 Contenido del Notebook

### **Objetivo**
Justificar y validar **Swin Transformer Small (multiseed)** como modelo elegido para clasificación de ACL.

### **Fases**

1. **Phase 1: Setup & Configuration**
   - Importaciones y configuración de paths
   - Definición de constantes y funciones utilitarias

2. **Phase 2 & 3: Baselines & Comparison Table**
   - Carga de resultados de todos los modelos base
   - Tabla comparativa de métricas en validación (AUC, F1, Precision, Recall, Specificity)
   - Per-plane AUC breakdown

3. **Phase 4: Justificación Narrativa**
   - Análisis comparativo: ¿Por qué Swin Small?
   - Comparación vs Swin Base, ViT, CNN
   - Análisis de per-plane consistency
   - Beneficios de multiseed aggregation

4. **Phase 5-8: Model Loading, Validation & Test**
   - Carga de modelos Swin Small multiseed
   - Generación de predicciones ensemble weighted
   - Threshold tuning (recall-maximizing, precision ≥ 0.75)
   - Evaluación en test set
   - Validación externa con dataset de Croacia

## 🚀 Cómo Ejecutar

```bash
# Abrir VS Code y ejecutar el notebook
code /home/palodo2/tfg/acl_classifier/final_model/Swin_Small_Model_Selection_Validation.ipynb

# Ejecutar celdas en orden (Shift+Enter)
```

## 📊 Outputs Esperados

- ✅ Tabla CSV con comparación de modelos
- ✅ Gráficas de threshold tuning y resultados en test
- ✅ Análisis de validación externa (Croatia)
- ✅ Reporte de justificación del modelo seleccionado

## 🎯 Justificación Suin Small

### ✓ Ventajas Clave
1. **AUC Superior**: Compite con modelos más complejos
2. **Balance Clínico**: Recall alto (minimiza falsos negativos)
3. **Consistencia Per-Plano**: Std bajo entre sagittal/coronal/axial
4. **Eficiencia**: Menos parámetros que Swin Base / ViT
5. **Reproducibilidad**: Multiseed (10 semillas) vs single-seed baseline

### ⚠️ Trade-offs
- Swin Base: +0.35% AUC pero +70% parámetros → No justificado
- ViT Base: -1% AUC con más complejidad → No competitivo
- CNN: -2.5% AUC, menos apto para medical imaging → No preferido

## 📝 Para el TFG

Puedes usar este notebook como base para:
1. **Tabla de justificación**: Mostrar comparativa de modelos en validación
2. **Narrative**: Explicación de por qué Swin Small es la mejor opción
3. **Resultados**: Métricas en test y validación externa
4. **Visualizaciones**: Gráficas de ROC, confusion matrices, threshold optimization

## 🔗 Referencias

- Modelos base: `/checkpoints/`
- Datos: `/data/`
- Código fuente: `/src/`
- Resultados: `/final_model/results/`

---

**Created**: 2026-05-05  
**Status**: Ready for TFG documentation
