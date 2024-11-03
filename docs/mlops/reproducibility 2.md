## Reproducibilidad del experimento de clasificación de emociones en la música turca

### Entorno de trabajo
* **Software:** Python 3.12, scikit-learn

### Datos
* **Fuente de los datos:** UC Irvine Machine Learnig Repository

https://archive.ics.uci.edu/dataset/862/turkish+music+emotion

* **Formato de los datos:** Tabular, csv.

### Código
* **Repositorio:** https://github.com/isak792/mlops_team_20/


### Entrenamiento del modelo
* **Hiperparámetros:**

    Random Forest: n_estimators: 50, max_depth: 5

    SVM: C: 1.0, kernel: 'rbf', gamma: 'scale'

    KNN: n_neighbors: 5

### Evaluación del modelo
* **Métricas:** Accuracy, F1-score, recall y precision.
* **Resultados:** Accuracy de 92% en el conjunto de prueba
"""