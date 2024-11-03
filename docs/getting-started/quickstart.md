# Guía de instalación rápida

# Requisitos del sistema

- Python 3.12
- pip
- make
- Librerías:
    * pandas
    * numpy
    * scikit-learn
    * matplotlib


# Pasos a seguir

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
3. Configura el entorno de Desarrollo e instala dependencias

  ```
  make setup
  ```

4. Para ejecutar el proyecto, usa:
  
  ```bash
  python run_pipeline.py
  ```
"""