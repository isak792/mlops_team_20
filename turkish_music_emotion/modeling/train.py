import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import yaml
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, data_path: Path, model_output_path: Path):
        self.data_path = data_path
        self.model_output_path = model_output_path
        self.model = None
        self.metrics = {}

    def load_data(self):
        logger.info(f"Loading data for training from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        logger.info("Training data loaded successfully.")
        return self.data

    def train_model(self):
        data = self.load_data()
        X = data.drop(columns=['Class'])
        y = data['Class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        
        self.model = KNeighborsClassifier()
        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)

        self.metrics['accuracy'] = accuracy_score(y_test, predictions)
        self.metrics['precision'] = precision_score(y_test, predictions, average='weighted')
        self.metrics['recall'] = recall_score(y_test, predictions, average='weighted')
        self.metrics['f1_score'] = f1_score(y_test, predictions, average='weighted')
        self.metrics['confusion_matrix'] = confusion_matrix(y_test, predictions).tolist()

        logger.info(f"Model trained successfully with metrics: {self.metrics}")

    def save_metrics(self):
        with open(Path(self.model_output_path).parent / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f)
        logger.info("Metrics saved successfully.")

    def save_model(self):
        if self.model is None:
            logger.warning("No model to save. Train the model first.")
            return
        
        joblib.dump(self.model, self.model_output_path)
        logger.info(f"Model saved successfully to {self.model_output_path}") 

    def run(self):
        self.train_model()
        self.save_metrics()
        self.save_model()


if __name__ == "__main__":
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    trainer = ModelTrainer(
        PROCESSED_DATA_DIR / params['data']['output_filename'],
        MODELS_DIR / params['model']['output_filename']
    )
    trainer.run()
