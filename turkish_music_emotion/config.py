"""
Módulo de configuración del proyecto que establece las rutas del sistema de archivos
y configura el logging.

Este módulo inicializa las rutas principales del proyecto siguiendo una estructura
de directorio estándar para proyectos de ciencia de datos. También configura
el sistema de logging usando loguru, con soporte opcional para tqdm.

Attributes:
    PROJ_ROOT (Path): Ruta raíz del proyecto, calculada dinámicamente.
    DATA_DIR (Path): Directorio principal de datos.
    RAW_DATA_DIR (Path): Directorio para datos sin procesar.
    INTERIM_DATA_DIR (Path): Directorio para datos en proceso intermedio.
    PROCESSED_DATA_DIR (Path): Directorio para datos procesados finales.
    EXTERNAL_DATA_DIR (Path): Directorio para datos externos.
    MODELS_DIR (Path): Directorio para almacenar modelos entrenados.
    REPORTS_DIR (Path): Directorio para reportes y documentación.
    FIGURES_DIR (Path): Directorio para gráficos y visualizaciones.

Note:
    Este módulo asume una estructura de proyecto específica:
    ```
    proyecto/
    ├── data/
    │   ├── raw/
    │   ├── interim/
    │   ├── processed/
    │   └── external/
    ├── models/
    └── reports/
        └── figures/
    ```

Example:
    Para usar las rutas definidas en otros módulos:
    >>> from config import RAW_DATA_DIR
    >>> data_file = RAW_DATA_DIR / \"dataset.csv\"
    >>> print(data_file)
"""

from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Cargar variables de entorno desde archivo .env si existe
load_dotenv()

# Definición de rutas del proyecto
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Estructura de directorios para datos
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Directorios para modelos y reportes
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Configuración de loguru con soporte para tqdm si está instalado
# Ver: https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm
    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
