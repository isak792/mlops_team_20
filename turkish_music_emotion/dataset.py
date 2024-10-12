from pathlib import Path
from loguru import logger
import pandas as pd
import yaml

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


if __name__ == "__main__":
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    handler = DataHandler()
    handler.run(RAW_DATA_DIR, params['data']['input_filename'], params['data']['output_filename'])