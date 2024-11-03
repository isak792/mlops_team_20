"""
Módulo para realizar predicciones utilizando modelos entrenados de machine learning.

Este módulo proporciona una interfaz para cargar modelos previamente entrenados,
junto con sus label encoders y realizar predicciones sobre nuevos datos.
"""

import joblib
import json
import pandas as pd
import yaml
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

class ModelPredictor:
    """
    Clase para realizar predicciones usando modelos entrenados.

    Esta clase maneja la carga de modelos previamente entrenados, sus label encoders
    y los índices de test, permitiendo realizar predicciones sobre nuevos datos de
    manera consistente con el entrenamiento.

    Attributes:
        model_path (Path): Ruta al archivo del modelo guardado.
        le_path (Path): Ruta al archivo del LabelEncoder guardado.
        split_indices_path (Path): Ruta al archivo con los índices de train/test.
        model (BaseEstimator): Modelo cargado (inicialmente None).
        le (LabelEncoder): LabelEncoder cargado (inicialmente None).
        test_indices (list): Índices del conjunto de test (inicialmente None).

    Example:
        >>> predictor = ModelPredictor(
        ...     model_path=Path("models/model.joblib"),
        ...     le_path=Path("models/label_encoder.joblib"),
        ...     split_indices_path=Path("models/split_indices.json")
        ... )
        >>> predictions = predictor.predict(input_data)
    """

    def __init__(self, model_path: Path, le_path: Path, split_indices_path: Path):
        """
        Inicializa el predictor con las rutas necesarias.

        Args:
            model_path (Path): Ruta al archivo del modelo guardado.
            le_path (Path): Ruta al archivo del LabelEncoder guardado.
            split_indices_path (Path): Ruta al archivo con índices de train/test.
        """
        self.model_path = model_path
        self.le_path = le_path
        self.split_indices_path = split_indices_path
        self.model = None
        self.le = None
        self.test_indices = None

    def load_model(self) -> None:
        """
        Carga el modelo desde el archivo guardado.

        Raises:
            FileNotFoundError: Si no se encuentra el archivo del modelo.
            Exception: Si hay errores al cargar el modelo.
        """
        self.model: BaseEstimator = joblib.load(self.model_path)

    def load_label_encoder(self) -> None:
        """
        Carga el LabelEncoder desde el archivo guardado.

        Raises:
            FileNotFoundError: Si no se encuentra el archivo del LabelEncoder.
            Exception: Si hay errores al cargar el LabelEncoder.
        """
        self.le: LabelEncoder = joblib.load(self.le_path)

    def load_test_indices(self) -> None:
        """
        Carga los índices del conjunto de test desde el archivo JSON.

        Raises:
            FileNotFoundError: Si no se encuentra el archivo de índices.
            JSONDecodeError: Si el archivo no tiene un formato JSON válido.
        """
        with open(self.split_indices_path, 'r') as f:
            split_indices = json.load(f)
        self.test_indices = split_indices['test_indices']

    def predict(self, X: pd.DataFrame) -> list:
        """
        Realiza predicciones sobre nuevos datos.

        Esta función carga automáticamente el modelo, label encoder e índices
        si no han sido cargados previamente.

        Args:
            X (pd.DataFrame): DataFrame con los datos para predecir.

        Returns:
            list: Lista con las predicciones en formato de etiquetas originales.

        Raises:
            ValueError: Si el DataFrame no contiene las columnas esperadas.
            Exception: Si hay errores durante la predicción.

        Example:
            >>> input_data = pd.read_csv("new_data.csv")
            >>> predictions = predictor.predict(input_data)
            >>> print(predictions[:5])
            ['happy', 'sad', 'angry', 'relax', 'happy']
        """
        if self.model is None:
            self.load_model()
        if self.le is None:
            self.load_label_encoder()
        if self.test_indices is None:
            self.load_test_indices()
        
        # Verifica si el DataFrame de entrada está vacío
        if X.empty:
           raise ValueError("El DataFrame de entrada está vacío. No se pueden hacer predicciones.")

        X_test = X.loc[self.test_indices]
        predictions = self.model.predict(X_test)
        return self.le.inverse_transform(predictions).tolist()


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Carga los datos de entrada desde un archivo CSV.

    Args:
        data_path (Path): Ruta al archivo CSV con los datos.

    Returns:
        pd.DataFrame: DataFrame con los features (sin la columna 'Class').

    Raises:
        FileNotFoundError: Si no se encuentra el archivo de datos.
        pd.errors.EmptyDataError: Si el archivo está vacío.
    """
    data = pd.read_csv(data_path)
    return data.drop(columns=['Class'])

def load_config(config_path: Path = Path('params.yaml')) -> dict:
    """
    Carga la configuración desde un archivo YAML.

    Args:
        config_path (Path, opcional): Ruta al archivo de configuración.
                                    Defaults to Path('params.yaml').

    Returns:
        dict: Diccionario con la configuración cargada.

    Raises:
        FileNotFoundError: Si no se encuentra el archivo de configuración.
        yaml.YAMLError: Si hay errores en el formato del archivo YAML.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config()
    
    PROCESSED_DATA_DIR = Path(config['data']['processed_dir'])
    MODELS_DIR = Path(config['model']['models_dir'])

    predictor = ModelPredictor(
        MODELS_DIR / config['model']['output_filename'],
        MODELS_DIR / config['model']['le_filename'],
        MODELS_DIR / config['model']['split_indices_filename']
    )

    input_data = load_data(PROCESSED_DATA_DIR / config['data']['processed_filename'])
    predictions = predictor.predict(input_data)
    print(f"Predictions for test set: {predictions}")
