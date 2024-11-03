## Preprocesamiento de datos

### Exploración de los datos
* **Histogramas:** Se observó una distribución uniforme para la distribución de las cuatro clases.
* **Valores faltantes:** El dataset no contiene valores faltantes.

### Transformación de datos
* **Manejo de valores atípicos:** outliers fueron manejados por clase utilizando límites de percentiles.
* **Selección de características:** Se eliminaron variables altamente correlacionadas y se seleccionaron las basadas en información mutua con la variable objetivo
* **Codificación:** Se utilizó one-hot encoding para codificar la variable categórica 'class'.

### División de los datos
* **Conjunto de entrenamiento:** 80%
* **Conjunto de prueba:** 20%
"""