import joblib
import pandas as pd
import numpy as np
import json
import yaml
import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, data_path: Path, model_output_path: Path, le_output_path: Path, split_indices_path: Path, model_type: str = 'knn', hyperparameters: dict = None):
        self.data_path = data_path
        self.model_output_path = model_output_path
        self.le_output_path = le_output_path
        self.split_indices_path = split_indices_path
        self.metrics_output_path = model_output_path.parent / 'metrics.json'
        self.model_type = model_type
        self.hyperparameters = hyperparameters or {}
        self.model = None
        self.metrics = {}
        self.le = LabelEncoder()

        mlflow.set_tracking_uri("http://24.144.69.175:5000")
        mlflow.set_experiment(f"/dvc_pipe/testing")

    def load_data(self) -> pd.DataFrame:
        """Load data from a CSV file."""
        logger.info(f"Loading data for training from {self.data_path}")
        data = pd.read_csv(self.data_path)
        logger.info("Training data loaded successfully.")
        return data

    def preprocess_data(self, data: pd.DataFrame, target_column: str = 'Class') -> Tuple[pd.DataFrame, np.ndarray]:
        """Preprocess the data by splitting features and target, and encoding the target."""
        X = data.drop(columns=[target_column])
        y = data[target_column]
        y_encoded = self.le.fit_transform(y)
        return X, y_encoded

    def split_data(self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, random_state: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Split the data into training and testing sets."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        split_indices = {
            'train_indices': X_train.index.tolist(),
            'test_indices': X_test.index.tolist()
        }

        with open(self.split_indices_path, 'w') as f:
            json.dump(split_indices, f)
        
        logger.info(f"Split indices saved to {self.split_indices_path}")
        return X_train, X_test, y_train, y_test

    def create_model(self) -> BaseEstimator:
        """Create and return the specified model."""
        if self.model_type == 'knn':
            return KNeighborsClassifier(**self.hyperparameters.get('knn', {}))
        elif self.model_type == 'rf':
            return RandomForestClassifier(**self.hyperparameters.get('rf', {}))
        elif self.model_type == 'svm':
            return SVC(**self.hyperparameters.get('svm', {}))
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train_model(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """Train the model using the specified algorithm."""
        self.model = self.create_model()

        mlflow.log_param("model_type", self.model_type)
        if self.model_type == 'rf':
            mlflow.log_param("n_estimators", self.hyperparameters.get('n_estimators', 100))
            mlflow.log_param("max_depth", self.hyperparameters.get('max_depth', None))
        elif self.model_type == 'svm':
            mlflow.log_param("C", self.hyperparameters.get('C', 1.0))
            mlflow.log_param("kernel", self.hyperparameters.get('kernel', 'rbf'))
            mlflow.log_param("gamma", self.hyperparameters.get('gamma', 'scale'))
        elif self.model_type == 'knn':
            mlflow.log_param("n_neighbors", self.hyperparameters.get('n_neighbors', 5))

        self.model.fit(X_train, y_train)
        logger.info(f"Model trained successfully.")

        mlflow.sklearn.log_model(self.model, "model")
        logger.info("Model logged to MLflow.")

    def evaluate_model(self, X_test: pd.DataFrame, y_test: np.ndarray) -> None:
        """Evaluate the model and store performance metrics including confusion matrix."""
        predictions = self.model.predict(X_test)
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, predictions, average='weighted', zero_division=0)
        }
        
        cm = confusion_matrix(y_test, predictions)
        mlflow.log_metrics(self.metrics)
        cm_dict = {
            'confusion_matrix': cm.tolist()
        }
        mlflow.log_dict(cm_dict, "confusion_matrix.json")
        
        logger.info("Metrics and confusion matrix logged to MLflow.")
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

    def save_label_encoder(self) -> None:
        """Save the label encoder to a file."""
        joblib.dump(self.le, self.le_output_path)
        logger.info(f"Label encoder saved successfully to {self.le_output_path}")

    def run(self) -> None:
        """Run the entire model training pipeline."""
        with mlflow.start_run():
            data = self.load_data()
            X, y_encoded = self.preprocess_data(data)
            X_train, X_test, y_train, y_test = self.split_data(X, y_encoded)

            self.train_model(X_train, y_train)
            self.evaluate_model(X_test, y_test)
            
            self.save_metrics()
            self.save_model()
            self.save_label_encoder()

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config(Path('params.yaml'))
    
    PROCESSED_DATA_DIR = Path(config['data']['processed_dir'])
    MODELS_DIR = Path(config['model']['models_dir'])
    
    trainer = ModelTrainer(
        PROCESSED_DATA_DIR / config['data']['processed_filename'],
        MODELS_DIR / config['model']['output_filename'],
        MODELS_DIR / config['model']['le_filename'],
        MODELS_DIR / config['model']['split_indices_filename'],
        model_type=config['model']['type']
    )
    trainer.run()
