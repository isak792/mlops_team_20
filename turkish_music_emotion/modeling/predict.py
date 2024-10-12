import joblib
import json
import pandas as pd
import yaml
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

class ModelPredictor:
    def __init__(self, model_path: Path, le_path: Path, split_indices_path: Path):
        self.model_path = model_path
        self.le_path = le_path
        self.split_indices_path = split_indices_path
        self.model = None
        self.le = None
        self.test_indices = None

    def load_model(self):
        self.model: BaseEstimator = joblib.load(self.model_path)

    def load_label_encoder(self):
        self.le: LabelEncoder = joblib.load(self.le_path)

    def load_test_indices(self):
        with open(self.split_indices_path, 'r') as f:
            split_indices = json.load(f)
        self.test_indices = split_indices['test_indices']

    def predict(self, X: pd.DataFrame) -> list:
        if self.model is None:
            self.load_model()
        if self.le is None:
            self.load_label_encoder()
        if self.test_indices is None:
            self.load_test_indices()

        X_test = X.loc[self.test_indices]
        predictions = self.model.predict(X_test)
        return self.le.inverse_transform(predictions).tolist()

def load_data(data_path: Path) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    return data.drop(columns=['Class'])  

def load_config(config_path: Path = Path('params.yaml')) -> dict:
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
