# Revisión 2 de la memoria — seguimiento

## Formato y estructura
- [x] 1. "Cuadro" → "Tabla" en todo el documento (renewcommand tablename)
- [x] 2. Captions/títulos: indicar (validación) o (test)
- [x] 3. Juntar cap. 2 (Materiales) + cap. 3 (Métodos) → "Materiales y Métodos"
- [x] 4. Incluir el material complementario (anexo) en el cuerpo (3 apartados)
- [x] 5. Métodos: añadir 3.8 (experimentación por modelo + métricas)
- [x] 6. Métodos: añadir 3.9 (paquetes/versiones/hardware, del anexo)
- [x] 7. Reorganizar Resultados: 4.1 ResNet · 4.2 ViT · 4.3 Swin · 4.4 Comparativa y análisis · 4.5 Validación (4.5.1 zero-shot, 4.5.2 fine-tuning) · 4.6 Efecto del tamaño · 4.7 Discusión final
- [x] 7b. 4.0.1 (diseño experimental) → mover a Métodos
- [x] 7c. Dentro de 4.1/4.2/4.3: "Rendimiento por vista (multi-semilla + checkpoint)" y "Ensamble multi-vista calibrado"

## Comentarios puntuales
- [x] 8. Esquema del Swin Transformer (figura en métodos)
- [x] 9. Pág 25 (materiales): imagen de cada plano como ejemplo
- [x] 10. 3.2.1: cómo se hizo la exploración y por qué la config final
- [x] 11. Tablas 3.2/3.3: cómo se obtuvieron los valores (grid/búsqueda)
- [x] 12. Justificar pooling: attention en sag/axial, max en coronal
- [x] 13. Tabla 4.1 (estabilidad multi-semilla) → boxplot
- [x] 14. Comentar las gráficas de "overfitting analysis" (todos los modelos)
- [x] 15. Matrices de confusión gráficas en vez de tablas (por modelo; Croacia se mantiene en tabla)
- [x] 16. Pág 39: arreglar primera frase (menciona Swin antes de introducirlo)
- [x] 17. Fig 4.9: rodear zona de interés / añadir Grad-CAM
- [x] 18. 5.1 (conclusiones): esquema de la arquitectura

## Build
- Compilar con `tectonic` (no hay pdflatex/latexmk): `export PATH=/home/palodo2/bin:$PATH && tectonic -X compile ejemplo-memoria.tex`.

## Revisión 3 (tutores)
- [x] R1. Portada: separar nombre del autor de los tutores (vspace cls 0.5→1.6; +4pt entre tutores)
- [x] R2. Apóstrofes del resum valenciano: `'` activo en babel-spanish → sustituido por ’ (U+2019)
- [x] R3. Secciones 4.0.* (\subsection en lugar de \section): ya resuelto en la reorg de resultados; limpiada cabecera obsoleta
- [x] R4. Quitar "(obligatorio)" (era solo comentario LaTeX del resumen español)
- [x] R5. Sin sangría tras ecuación cuando continúa la frase: `\noindent` en materiales (1) y metodología (3)
- [x] R6. Mención breve de prueba sobre rodillas propias en Validación externa (redacción cualitativa, falta confirmar resultado real)
- [x] R7. Enlace a GitHub (código + memoria + app): footnote con \url{https://github.com/palodo/TFG-ACL} en el ítem "Demostrador interactivo" de conclusiones
