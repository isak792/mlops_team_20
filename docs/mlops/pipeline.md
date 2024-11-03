## Pipeline del sistema de clasificación de emociones en la música turca

### Descripción detallada de cada etapa
* **Datos de entrada:** Los datos se descargaron de la siguiente ubicación:

  https://archive.ics.uci.edu/dataset/862/turkish+music+emotion

* **Preprocesamiento:** Se aplica una transformación one-hot encoding para codificar la variable categórica 'class' y se utiliza una combinación de técnicas para seleccionar las características más relevantes. Se manejan valores atípicos.
* **Entrenamiento:** Se puede entrenar 3 diferentes modelos: random forest (rf), algoritmo vecino k-más cercano (knn) y máquina de vectores de soporte(svm).
* **Evaluación:** Se evalúa el modelo utilizando las métricas Accuracy, f1 scores, recall y precision utilizando Mlflow como herramienta para guardar los experimentos y las métricas, así como los modelos seleccionados.

### Principales Herramientas y tecnologías
- **Data Ingestion**: Python, pandas
- **Preprocessing**: scikit-learn, pandas
- **Model Training**: scikit-learn
- **MLOps**: MLflow, Docker, DVC
"""