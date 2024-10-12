import json
import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
import yaml
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, data_path: Path, model_output_path: Path, model_type: str = 'knn'):
        self.data_path = data_path
        self.model_output_path = model_output_path
        self.metrics_output_path = model_output_path.parent / 'metrics.json'
        self.model_type = model_type
        self.model = None
        self.metrics = {}
        self.le = LabelEncoder()

    def load_data(self) -> pd.DataFrame:
        """Load data from a CSV file."""
        logger.info(f"Loading data for training from {self.data_path}")
        data = pd.read_csv(self.data_path)
        logger.info("Training data loaded successfully.")
        return data

    def preprocess_data(self, data: pd.DataFrame, target_column: str = 'Class') -> tuple:
        """Preprocess the data by splitting features and target, and encoding the target."""
        X = data.drop(columns=[target_column])
        y = data[target_column]
        y_encoded = self.le.fit_transform(y)
        return X, y_encoded

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 1) -> tuple:
        """Split the data into training and testing sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def create_model(self) -> BaseEstimator:
        """Create and return the specified model."""
        if self.model_type == 'knn':
            return KNeighborsClassifier()
        elif self.model_type == 'rf':
            return RandomForestClassifier()
        elif self.model_type == 'svm':
            return SVC()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model using the specified algorithm."""
        self.model = self.create_model()
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Evaluate the model and store performance metrics."""
        predictions = self.model.predict(X_test)
        self.metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1_score': f1_score(y_test, predictions, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, predictions).tolist()
        }
        logger.info(f"Model evaluated successfully with metrics: {self.metrics}")

    def save_metrics(self) -> None:
        """Save metrics to a JSON file."""
        with open(self.metrics_output_path, 'w') as f:
            json.dump(self.metrics, f)
        logger.info(f"Metrics saved successfully to {self.metrics_output_path}")

    def save_model(self) -> None:
        """Save the trained model to a file."""
        joblib.dump(self.model, self.model_output_path)
        logger.info(f"Model saved successfully to {self.model_output_path}")

    def run(self) -> None:
        """Run the entire model training pipeline."""
        data = self.load_data()
        X, y_encoded = self.preprocess_data(data)
        X_train, X_test, y_train, y_test = self.split_data(X, y_encoded)
        
        self.train_model(X_train, y_train)
        self.evaluate_model(X_test, y_test)
        
        self.save_metrics()
        self.save_model()

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config(Path('params.yaml'))
    
    PROCESSED_DATA_DIR = Path(config['data']['processed_dir'])
    MODELS_DIR = Path(config['model']['models_dir'])
    
    trainer = ModelTrainer(
        PROCESSED_DATA_DIR / config['data']['output_filename'],
        MODELS_DIR / config['model']['output_filename'],
        model_type=config['model']['type']
    )
    trainer.run()
