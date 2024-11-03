## Proceso de inferencia del modelo de clasificación de imágenes

### Preparación de los datos de entrada
Los datos de entrada se encuentran en un archivo .csv y siguen el proceso de selección de características, manejo de valores atípicos y encoding a la variable categórica.

### Carga del modelo
Se utilizan modelos de la libreria scikit-learn: random forest, svm y knn.

### Proceso de inferencia
1. Los datos se preprocesan según lo descrito anteriormente.
2. Los datos procesados se pasan como entrada al modelo.
3. El modelo genera una probabilidad para cada clase.
4. Se selecciona la clase con la mayor probabilidad como la predicción final.

### Evaluación del modelo
El modelo se evalúa utilizando mlflow como herramienta de versionado para los experimentos y generación del modelo.
"""