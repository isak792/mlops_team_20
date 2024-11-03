# Descripción general del proyecto

Este repositorio contiene un proyecto de MLOps que implementa un pipeline completo de Machine Learning, desde la gestión y preprocesamiento de datos hasta el despliegue y automatización de modelos. Sigue los principios de MLOps para garantizar la reproducibilidad, escalabilidad y mantenimiento en producción.

Se está usando el dataset Turkish Music Emotion el cual contiene canciones turcas etiquetadas con cuatro diferentes emociones: alegría, tristeza, ira y relajacion. Este dataset considera un total de 100 piezas musicales para cada clase, de manera que, se tiene el mismo número de muestras para cada clase. Entre las características que se encuentran en el dataset están el tempo, la tonalidad, el ritmo, y otras características acústicas.

La princiapal aplicación de este proyecto es la clasificación de emociones en la música.
Este tipo de investigaciones permite crear sistemas de recomendación musical basados en el estado de ánimo y tiene aplicaciones interesantes en la musicoterapia y la inteligencia artificial para la música.


Referencia:
Bilal Er, M., & Aydilek, I. B. (2019). Music emotion recognition by using chroma spectrogram and deep visual features. Journal of Computational Intelligent Systems, 12(2), 1622–1634. International Journal of Computational Intelligence Systems, DOI: https://doi.org/10.2991/ijcis.d.191216.001 '''

# Requisitos del sistema

- Python 3.12
- pip
- make
- Librerías:
    * pandas
    * numpy
    * scikit-learn


# Guía de instalación rápida

1. Clona este repositorio:
   ```
   git clone https://github.com/isak792/mlops_team_20.git
   cd mlops_team_20
   ```
   
2. Crea y activa el entorno virtual, e instala las dependencias:

   ```bash
   python3 -m venv mlops_tme_venv
   ```

   Para dispositivos Windows, el código para activar el ambiente es el siguiente:
   ```bash
   mlops_tme_venv\Scripts\activate
   ```

   Para dispositivos MacOS y Linux:
   ```bash
   source mlops_tme_venv/bin/activate
   ```

# Configuración del Entorno de Desarrollo y Dependencias

El siguiente comando se utiliza para instalar todas las dependencias.
Además, este proyecto utiliza pre-commit hooks para mantener la calidad del código. Para configurar el entorno de desarrollo:

```
make setup
```

Este comando instalará pre-commit y configurará los hooks necesarios.

# Organización del Proyecto

```
├── LICENSE
├── Makefile                <- Makefile con comandos como `make data` o `make train`
├── README.md               <- El README de nivel superior para desarrolladores que usan este proyecto.
├── data
│   ├── external            <- Datos de fuentes de terceros.
│   ├── interim             <- Datos intermedios que han sido transformados.
│   ├── processed           <- Los conjuntos de datos finales, canónicos para modelado.
│   └── raw                 <- El volcado de datos original e inmutable.
│
├── docs                    <- Un proyecto mkdocs por defecto; ver www.mkdocs.org para detalles
│
├── models                  <- Modelos entrenados y serializados, predicciones de modelos o resúmenes de modelos
│
├── notebooks               <- Jupyter notebooks. La convención de nomenclatura es un número (para ordenar),
│                         las iniciales del creador y una breve descripción delimitada por `-`, p.ej.
│                         `1.0-jqp-exploracion-inicial-de-datos`.
│
├── references              <- Diccionarios de datos, manuales y todos los demás materiales explicativos.
│
├── reports                 <- Análisis generado como HTML, PDF, LaTeX, etc.
│   └── figures             <- Gráficos y figuras generadas para ser usadas en informes
│
├── requirements.txt        <- El archivo de requisitos para reproducir el entorno de análisis,
│                         generado con `pip freeze > requirements.txt`
│
├── setup.cfg               <- Archivo de configuración para flake8
├── pyproject.toml          <- Archivo de configuración del proyecto
│
├── .pre-commit-config.yaml <- Configuración para pre-commit hooks
│
│── turkish_music_emotion   <- Código fuente para uso en este proyecto.
│   ├── __init__.py         <- Hace que mlops sea un módulo Python
│   ├── config.py           <- Almacena variables útiles y configuración
│   ├── dataset.py          <- Scripts para descargar o generar datos
│   ├── features.py         <- Código para crear características para modelado
│   ├── modeling
│   │   ├── __init__.py
│   │   ├── predict.py      <- Código para ejecutar inferencia de modelos con modelos entrenados
│   │   └── train.py        <- Código para entrenar modelos
│   └── plots.py            <- Código para crear visualizaciones
│
│── dvc.yaml                <- Configuración del Pipeline DVC
│
│── params.yaml             <- Parámetros utilizados por el pipeline
│
└── run_pipeline.py         <- Archivo que define los pasos a seguir por el pipeline
```

# Calidad del Código

Este proyecto utiliza las siguientes herramientas para mantener la calidad del código:

- Black: Para formateo de código
- isort: Para ordenar las importaciones
- Flake8: Para hacer cumplir la guía de estilo

Estas herramientas se ejecutan automáticamente en cada commit a través de pre-commit hooks.

Para ejecutar estas herramientas manualmente, puedes usar:

```
make lint

# Ejemplo básico de uso

Carga y despliegue de los datos de entrada originales

from turkish_music_emotion.dataset import DataHandler

df = dh.load_data('/data/raw/Acoustic Features.csv')

df
"""