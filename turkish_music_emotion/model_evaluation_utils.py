"""
Script de utilidades para evaluación de modelos de machine learning.

Este módulo proporciona herramientas para cargar, evaluar y visualizar resultados
de modelos de clasificación de emociones en música, integrándose con MLflow.
"""

import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from urllib.parse import urlparse

class ModelEvaluationUtils:
    """
    Clase para evaluar y visualizar resultados de modelos de clasificación de emociones.

    Esta clase proporciona métodos para cargar modelos desde MLflow, procesar datos,
    realizar predicciones y generar métricas y visualizaciones de evaluación.

    Attributes:
        model_uri (str): URI del modelo en MLflow.
        data_path (str): Ruta al archivo de datos para evaluación.
        model: Modelo cargado desde MLflow.
        data (pd.DataFrame): DataFrame con los datos de evaluación.
        label_map (dict): Mapeo de etiquetas numéricas a nombres de emociones.

    Example:
        >>> evaluator = ModelEvaluationUtils("mlflow://localhost:5000/1", "data/test.csv")
        >>> X, y = evaluator.prepare_data()
        >>> predictions = evaluator.make_predictions(X)
        >>> evaluator.plot_confusion_matrix(y, predictions)
    """

    def __init__(self, model_uri: str, data_path: str):
        """
        Inicializa el evaluador de modelos.

        Args:
            model_uri (str): URI del modelo en MLflow.
            data_path (str): Ruta al archivo de datos para evaluación.
        """
        self.model_uri = model_uri
        self.data_path = data_path
        self.model = self.load_model()
        self.data = self.load_data()
        self.label_map = {0: 'angry', 1: 'happy', 2: 'relax', 3: 'sad'}

    def load_model(self):
        """
        Carga el modelo desde MLflow usando el URI especificado.

        Returns:
            mlflow.pyfunc.PyFuncModel: Modelo cargado desde MLflow.

        Raises:
            mlflow.exceptions.MlflowException: Si hay problemas al cargar el modelo.
            
        Note:
            El URI debe seguir el formato mlflow://<host>:<port>/<run_id>
        """
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

    def load_data(self) -> pd.DataFrame:
        """
        Carga los datos de evaluación desde el archivo especificado.

        Returns:
            pd.DataFrame: DataFrame con los datos cargados.

        Raises:
            FileNotFoundError: Si no se encuentra el archivo de datos.
        """
        return pd.read_csv(self.data_path)
    
    def prepare_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepara los datos separando features y target.

        Returns:
            tuple[pd.DataFrame, pd.Series]: Features (X) y variable objetivo (y).
        """
        X = self.data.drop('Class', axis=1)
        y = self.data['Class']
        return X, y
    
    def make_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones y las convierte a etiquetas de emociones.

        Args:
            X (pd.DataFrame): Features para realizar predicciones.

        Returns:
            np.ndarray: Array con las predicciones convertidas a etiquetas de emociones.

        Example:
            >>> X, _ = evaluator.prepare_data()
            >>> predictions = evaluator.make_predictions(X)
            >>> print(predictions[:5])
            ['happy' 'sad' 'relax' 'angry' 'happy']
        """
        numeric_predictions = self.model.predict(X)
        return np.array([self.label_map[pred] for pred in numeric_predictions])

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Genera y muestra la matriz de confusión.

        Args:
            y_true (np.ndarray): Etiquetas verdaderas.
            y_pred (np.ndarray): Etiquetas predichas.

        Note:
            La matriz se muestra usando seaborn con un mapa de calor en tonos azules.
            Los valores se muestran en cada celda de la matriz.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
        """
        Calcula las métricas principales de evaluación.

        Args:
            y_true (np.ndarray): Etiquetas verdaderas.
            y_pred (np.ndarray): Etiquetas predichas.

        Returns:
            tuple[float, float, float, float]: Tupla con accuracy, precision, recall y F1-score.

        Example:
            >>> accuracy, precision, recall, f1 = evaluator.calculate_metrics(y_true, y_pred)
            >>> print(f"Accuracy: {accuracy:.2f}")
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        return accuracy, precision, recall, f1

    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Genera un reporte detallado de clasificación.

        Args:
            y_true (np.ndarray): Etiquetas verdaderas.
            y_pred (np.ndarray): Etiquetas predichas.

        Returns:
            str: Reporte de clasificación con métricas por clase.

        Example:
            >>> report = evaluator.generate_classification_report(y_true, y_pred)
            >>> print(report)
        """
        return classification_report(y_true, y_pred)
