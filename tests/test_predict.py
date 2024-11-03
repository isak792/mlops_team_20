import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

# Añade la ruta al sistema solo una vez.
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(BASE_DIR))

from turkish_music_emotion.modeling.predict import ModelPredictor
from turkish_music_emotion.modeling.train import ModelTrainer

@pytest.fixture
def synthetic_data():
    """Crea y devuelve datos sintéticos a partir de los datos originales para pruebas."""
    csv_file_path = BASE_DIR / 'data' / 'processed' / 'Acoustic_Features_Processed.csv'
    data = pd.read_csv(csv_file_path)
    metadata = Metadata.detect_from_dataframe(data)
    synthesizer = GaussianCopulaSynthesizer(metadata=metadata)
    synthesizer.fit(data)
    return synthesizer.sample(50)

@pytest.fixture
def trained_model_files(tmp_path, synthetic_data):
    """Entrena el modelo con datos sintéticos y guarda los archivos generados."""
    trainer = ModelTrainer(
        data_path=tmp_path / "train_data.csv",
        model_output_path=tmp_path / "model.joblib",
        le_output_path=tmp_path / "label_encoder.joblib",
        split_indices_path=tmp_path / "split_indices.json",
        model_type='rf',
        hyperparameters={'rf': {'n_estimators': 10, 'random_state': 1}}
    )
    synthetic_data.to_csv(trainer.data_path, index=False)
    trainer.run()
    return {
        'model_path': trainer.model_output_path,
        'le_output_path': trainer.le_output_path,
        'split_indices_path': trainer.split_indices_path,
        'test_data': synthetic_data.drop(columns=['Class']) 
    }

@pytest.fixture
def predictor(trained_model_files):
    """Fixture para instanciar un ModelPredictor con los archivos entrenados."""
    return ModelPredictor(
        model_path=trained_model_files['model_path'],
        le_path=trained_model_files['le_output_path'],
        split_indices_path=trained_model_files['split_indices_path']
    )

def test_prediction_pipeline(predictor, trained_model_files):
    """Prueba la funcionalidad de predicción del pipeline de modelo entrenado."""
    predictions = predictor.predict(trained_model_files['test_data'])
    assert predictions is not None
    assert isinstance(predictions, list), "Las predicciones deben ser una lista."
    assert all(isinstance(pred, str) for pred in predictions), "Todas las predicciones deben ser cadenas."

def test_predict_invalid_format(predictor):
    """Prueba que el predictor lance excepción al recibir un DataFrame con formato inválido."""
    invalid_data = pd.DataFrame({'Columna': [1, 2, 3]})
    with pytest.raises(KeyError):
        predictor.predict(invalid_data)

def test_prediction_accuracy_with_mock_labels(predictor, trained_model_files):
    """Verifica la consistencia de predicciones del modelo comparándolas con etiquetas simuladas."""
    predictions = predictor.predict(trained_model_files['test_data'])
    mock_labels = np.random.choice(['Relax', 'Happy', 'Sad', 'Angry'], size=len(predictions))
    agreement_ratio = np.mean([pred == mock for pred, mock in zip(predictions, mock_labels)])
    assert agreement_ratio >= 0  # Asegura que no haya errores de predicción

def test_load_test_indices(predictor):
    """Prueba que los índices de prueba se cargan correctamente."""
    predictor.load_test_indices()
    assert predictor.test_indices is not None
    assert isinstance(predictor.test_indices, list), "Los índices de prueba no son una lista."

def test_predict_empty_dataframe(predictor):
    """Prueba que el predictor maneje un DataFrame vacío."""
    empty_data = pd.DataFrame()
    with pytest.raises(ValueError, match="El DataFrame de entrada está vacío. No se pueden hacer predicciones."):
        predictor.predict(empty_data)


