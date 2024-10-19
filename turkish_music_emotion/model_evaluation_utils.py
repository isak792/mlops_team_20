import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from urllib.parse import urlparse

class ModelEvaluationUtils:
    def __init__(self, model_uri, data_path):
        self.model_uri = model_uri
        self.data_path = data_path
        self.model = self.load_model()
        self.data = self.load_data()
        self.label_map = {0: 'angry', 1: 'happy', 2: 'relax', 3: 'sad'}

    def load_model(self):
        parsed_uri = urlparse(self.model_uri)
        tracking_uri = f"{parsed_uri.scheme}://{parsed_uri.netloc}"
        print(f"Setting tracking URI to: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        
        fragment = parsed_uri.fragment
        run_id = fragment.split('/')[-1]
        print(f"Extracted run_id: {run_id}")
        
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        model_uri = f"runs:/{run_id}/model"
        print(f"Loading model from URI: {model_uri}")
        
        return mlflow.pyfunc.load_model(model_uri)    

    def load_data(self):
        return pd.read_csv(self.data_path)
    
    def prepare_data(self):
        X = self.data.drop('Class', axis=1)
        y = self.data['Class']
        return X, y
    
    def make_predictions(self, X):
        numeric_predictions = self.model.predict(X)
        return np.array([self.label_map[pred] for pred in numeric_predictions])

    def plot_confusion_matrix(self, y_true, y_pred):
        # Ensure y_true and y_pred are using string labels
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Get unique labels
        labels = sorted(set(y_true) | set(y_pred))

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def calculate_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        return accuracy, precision, recall, f1

    def generate_classification_report(self, y_true, y_pred):
        return classification_report(y_true, y_pred)
