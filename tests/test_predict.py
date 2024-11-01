import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from turkish_music_emotion.modeling.predict import ModelPredictor
from turkish_music_emotion.modeling.train import ModelTrainer

@pytest.fixture
def trained_model_files(tmp_path):

    csv_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed', 'Acoustic_Features_Processed.csv')
    
    data = pd.read_csv(csv_file_path)
    metadata = Metadata.detect_from_dataframe(data)
    synthesizer = GaussianCopulaSynthesizer(metadata=metadata)
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(50)
    
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
        'le_path': trainer.le_output_path,
        'split_indices_path': trainer.split_indices_path,
        'test_data': synthetic_data.drop(columns=['Class']) 
    }

# Se prueba la funcionalidad de la clase ModelPredictor para predecir con un modelo entrenado
def test_prediction_pipeline(trained_model_files):
    predictor = ModelPredictor(
        model_path=trained_model_files['model_path'],
        le_path=trained_model_files['le_path'],
        split_indices_path=trained_model_files['split_indices_path']
    )
    
    predictions = predictor.predict(trained_model_files['test_data'])
    
    assert predictions is not None
    assert isinstance(predictions, list)
    assert all(isinstance(pred, str) for pred in predictions)

# Se prueba que la clase ModelPredictor lance una excepción si se le pasa un DataFrame con un formato inválido
def test_predict_invalid_format(trained_model_files):
    predictor = ModelPredictor(
        model_path=trained_model_files['model_path'],
        le_path=trained_model_files['le_path'],
        split_indices_path=trained_model_files['split_indices_path']
    )
    
    invalid_data = pd.DataFrame({'Columna': [1, 2, 3]})
    with pytest.raises(KeyError):
        predictor.predict(invalid_data)