from pathlib import Path
from loguru import logger
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

from turkish_music_emotion.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

class DataHandler:
    def __init__(self):
        self.data = None

    def load_data(self, input_path: Path, input_filename: str):
        full_input_path = input_path / input_filename
        logger.info(f"Loading data from {full_input_path}")
        
        self.data = pd.read_csv(full_input_path)
        logger.success(f"Data loaded successfully from {full_input_path}")
        return self.data

    def save_data(self, output_dir: Path, filename: str):
        if self.data is None:
            logger.warning("No data to save. Load the data first.")
            return
        
        output_path = output_dir / filename
        logger.info(f"Saving data to {output_path}")
        
        self.data.to_csv(output_path, index=False)
        logger.success(f"Data saved successfully to {output_path}")

    def process_data(self):
        if self.data is None:
            logger.warning("No data loaded. Load data before processing.")
            return
        self.data.dropna(inplace=True)
        logger.success("Data processed successfully.")
        return self.data

    def run(self, input_path: Path, input_filename: str, output_filename: str):
        self.load_data(input_path, input_filename)
        self.process_data()
        self.save_data(PROCESSED_DATA_DIR, output_filename)



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


if __name__ == "__main__":
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    handler = DataHandler()
    handler.run(RAW_DATA_DIR, params['data']['input_filename'], params['data']['output_filename'])