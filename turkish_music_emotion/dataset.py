from pathlib import Path
from loguru import logger
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from turkish_music_emotion.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR

class DataHandler:
    def __init__(self):
        self.data = None
        self.input_path = None

    def load_raw_data(self, input_path: Path = RAW_DATA_DIR / "Acoustic Features.csv"):
        self.input_path = input_path
        logger.info(f"Loading raw data from {self.input_path}")
        self.data = pd.read_csv(self.input_path)
        logger.success("Raw data loaded successfully.")
        return self.data

    def load_processed_data(self, input_path: Path):
        self.input_path = input_path
        logger.info(f"Loading processed data from {self.input_path}")
        self.data = pd.read_csv(self.input_path)
        logger.success("Processed data loaded successfully.")
        return self.data

    def load_interim_data(self, input_path: Path):
        self.input_path = input_path
        logger.info(f"Loading interim data from {self.input_path}")
        self.data = pd.read_csv(self.input_path)
        logger.success("Interim data loaded successfully.")
        return self.data

    def save_processed_data(self, filename: str):
        if self.data is None:
            logger.warning("No data to save. Load the data first.")
            return
        
        output_path = PROCESSED_DATA_DIR / filename
        logger.info(f"Saving processed data to {output_path}")
        self.data.to_csv(output_path, index=False)
        logger.success("Processed data saved successfully.")

    def save_interim_data(self, filename: str):
        if self.data is None:
            logger.warning("No data to save. Load the data first.")
            return
        
        output_path = INTERIM_DATA_DIR / filename
        logger.info(f"Saving interim data to {output_path}")
        self.data.to_csv(output_path, index=False)
        logger.success("Interim data saved successfully.")

class MissingValueAnalyzer:
    def __init__(self, data):
        """Inicializa el objeto con el DataFrame."""
        self.data = data
    
    def get_missing_columns(self):
        """Obtiene las columnas con valores faltantes y sus respectivos conteos."""
        missing_values = self.data.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        return columns_with_missing
    
    def display_missing_columns(self):
        """Muestra las columnas con valores faltantes si existen."""
        columns_with_missing = self.get_missing_columns()
        if columns_with_missing.empty:
            print("No hay columnas con valores faltantes.")
        else:
            print("Columnas con valores faltantes:")
            print(columns_with_missing)


class LabelEncoderWrapper:
    def __init__(self, data):
        """Inicializa el objeto con el DataFrame."""
        self.data = data
        self.label_encoders = {}

    def encode_column(self, column):
        """Aplica label encoding a una columna categórica específica."""
        if column not in self.data.columns:
            raise ValueError(f"La columna '{column}' no existe en el DataFrame.")
        
        label_encoder = LabelEncoder()
        self.data[column] = label_encoder.fit_transform(self.data[column])
        self.label_encoders[column] = label_encoder  
        
    def get_encoded_data(self):
        """Devuelve el DataFrame con las columnas codificadas."""
        return self.data
    
    def decode_column(self, column):
        """Decodifica los valores en una columna previamente codificada."""
        if column not in self.label_encoders:
            raise ValueError(f"La columna '{column}' no fue codificada o no se ha guardado el codificador.")
        
        label_encoder = self.label_encoders[column]
        self.data[column] = label_encoder.inverse_transform(self.data[column])

