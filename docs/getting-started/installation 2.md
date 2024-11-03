# Instalación del proyecto

## Requisitos del sistema

- Python 3.12
- pip
- make
- Librerías:
    * pandas
    * numpy
    * scikit-learn
    * matplotlib
    * seaborn
    * mlflow
    * scipy
    * dvc


## Guía de instalación rápida

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

3. Configura el entorno de Desarrollo e instala dependencias:

  ```
  make setup
  ```

  o en su defecto instalar las dependencias mediante:

  pip install -r requirements.txt


4. Para ejecutar el proyecto, usa:
  
  ```bash
  python run_pipeline.py
  ```
"""