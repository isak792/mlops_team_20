import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

# Definir la ruta base solo una vez
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(BASE_DIR))

from turkish_music_emotion.modeling.train import ModelTrainer

@pytest.fixture
def synthetic_training_data():
    """Crea datos sintéticos a partir del archivo de características acústicas procesadas."""
    csv_file_path = BASE_DIR / 'data' / 'processed' / 'Acoustic_Features_Processed.csv'
    data = pd.read_csv(csv_file_path)
    metadata = Metadata.detect_from_dataframe(data)
    synthesizer = GaussianCopulaSynthesizer(metadata=metadata)
    synthesizer.fit(data)
    return synthesizer.sample(50)

@pytest.fixture
def trainer(tmp_path):
    """Instancia un ModelTrainer con configuraciones predeterminadas para las pruebas."""
    return ModelTrainer(
        data_path=tmp_path / "train_data.csv",
        model_output_path=tmp_path / "model.joblib",
        le_output_path=tmp_path / "label_encoder.joblib",
        split_indices_path=tmp_path / "split_indices.json",
        model_type='rf',
        hyperparameters={'rf': {'n_estimators': 10, 'random_state': 1}}
    )

def test_train_pipeline(trainer, synthetic_training_data, tmp_path):
    """Prueba el pipeline de entrenamiento completo del ModelTrainer."""
    synthetic_training_data.to_csv(trainer.data_path, index=False)
    
    trainer.run()
    
    assert trainer.model_output_path.exists(), "El archivo del modelo no fue generado."
    assert trainer.le_output_path.exists(), "El archivo de codificador de etiquetas no fue generado."
    assert trainer.split_indices_path.exists(), "El archivo de índices de división no fue generado."
    assert hasattr(trainer, 'metrics_output_path') and trainer.metrics_output_path.exists(), "El archivo de métricas no fue generado."

def test_model_types(trainer, synthetic_training_data):
    """Prueba el entrenamiento del ModelTrainer con distintos tipos de modelos."""
    synthetic_training_data.to_csv(trainer.data_path, index=False)
    
    for model_type in ['rf', 'knn', 'svm']:
        trainer.model_type = model_type
        trainer.run()
        assert trainer.model is not None, f"El modelo {model_type} no fue entrenado correctamente."

def test_evaluate_model(trainer, synthetic_training_data):
    """Prueba la evaluación del modelo con distintas métricas de rendimiento."""
    synthetic_training_data.to_csv(trainer.data_path, index=False)
    data = trainer.load_data()
    X, y_encoded = trainer.preprocess_data(data)
    X_train, X_test, y_train, y_test = trainer.split_data(X, y_encoded)
    trainer.train_model(X_train, y_train)

    trainer.evaluate_model(X_test, y_test)
    assert 'accuracy' in trainer.metrics, "La métrica de accuracy no está presente en las métricas."
    assert 'precision' in trainer.metrics, "La métrica de precision no está presente en las métricas."
    assert 'recall' in trainer.metrics, "La métrica de recall no está presente en las métricas."
    assert 'f1_score' in trainer.metrics, "La métrica de f1_score no está presente en las métricas."

    # Validar valores de métricas dentro de un rango razonable
    assert 0 <= trainer.metrics['accuracy'] <= 1, "Accuracy fuera de rango esperado."
    assert 0 <= trainer.metrics['precision'] <= 1, "Precision fuera de rango esperado."
    assert 0 <= trainer.metrics['recall'] <= 1, "Recall fuera de rango esperado."
    assert 0 <= trainer.metrics['f1_score'] <= 1, "F1 Score fuera de rango esperado."

def test_split_data_basic(trainer, synthetic_training_data):
    """Prueba la división de datos y verifica que los tamaños sean correctos."""
    # Guardar datos sintéticos en la ruta especificada
    synthetic_training_data.to_csv(trainer.data_path, index=False)
    
    # Cargar los datos
    data = trainer.load_data()
    
    # Preprocesar los datos
    X, y_encoded = trainer.preprocess_data(data)
    
    # Dividir los datos con un tamaño de prueba del 20%
    X_train, X_test, y_train, y_test = trainer.split_data(X, y_encoded, test_size=0.2, random_state=1)
    
    # Calcular el número esperado de ejemplos en los conjuntos de entrenamiento y prueba
    expected_train_size = int(len(X) * 0.8)  # 80% para entrenamiento
    expected_test_size = int(len(X) * 0.2)   # 20% para prueba
    
    # Comprobar tamaños
    assert len(X_train) == expected_train_size, f"El tamaño del conjunto de entrenamiento debería ser {expected_train_size}, pero fue {len(X_train)}."
    assert len(X_test) == expected_test_size, f"El tamaño del conjunto de prueba debería ser {expected_test_size}, pero fue {len(X_test)}."
    
    # Comprobar que los conjuntos no están vacíos
    assert len(X_train) > 0, "El conjunto de entrenamiento no debe estar vacío."
    assert len(X_test) > 0, "El conjunto de prueba no debe estar vacío."
    
    # Comprobar que las etiquetas también se dividen correctamente
    assert len(y_train) == len(X_train), "El tamaño de las etiquetas de entrenamiento no coincide con el tamaño de las características de entrenamiento."
    assert len(y_test) == len(X_test), "El tamaño de las etiquetas de prueba no coincide con el tamaño de las características de prueba."


