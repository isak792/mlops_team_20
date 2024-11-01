import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from turkish_music_emotion.modeling.train import ModelTrainer

@pytest.fixture
def synthetic_training_data():
    csv_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed', 'Acoustic_Features_Processed.csv')
    data = pd.read_csv(csv_file_path)
    metadata = Metadata.detect_from_dataframe(data)
    synthesizer = GaussianCopulaSynthesizer(metadata=metadata)
    synthesizer.fit(data)
    return synthesizer.sample(50)

@pytest.fixture
def trainer(tmp_path):
    return ModelTrainer(
        data_path=tmp_path / "train_data.csv",
        model_output_path=tmp_path / "model.joblib",
        le_output_path=tmp_path / "label_encoder.joblib",
        split_indices_path=tmp_path / "split_indices.json",
        model_type='rf',
        hyperparameters={'rf': {'n_estimators': 10, 'random_state': 1}}
    )

# Se prueba la funcionalidad de la clase ModelTrainer para entrenar un modelo
def test_train_pipeline(trainer, synthetic_training_data, tmp_path):
    synthetic_training_data.to_csv(trainer.data_path, index=False)
    
    trainer.run()
    
    assert trainer.model_output_path.exists()
    assert trainer.le_output_path.exists()
    assert trainer.split_indices_path.exists()
    assert trainer.metrics_output_path.exists()

# Se prueba la funcionalidad de la clase ModelTrainer para entrenar diferentes tipos de modelos
def test_model_types(trainer, synthetic_training_data):
    synthetic_training_data.to_csv(trainer.data_path, index=False)
    
    for model_type in ['rf', 'knn', 'svm']:
        trainer.model_type = model_type
        trainer.run()
        assert trainer.model is not None

# Se prueba la funcionalidad de la clase ModelTrainer para evaluar un modelo con diferentes métricas
def test_evaluate_model(trainer, synthetic_training_data):
    synthetic_training_data.to_csv(trainer.data_path, index=False)
    data = trainer.load_data()
    X, y_encoded = trainer.preprocess_data(data)
    X_train, X_test, y_train, y_test = trainer.split_data(X, y_encoded)
    trainer.train_model(X_train, y_train)

    trainer.evaluate_model(X_test, y_test)
    assert 'accuracy' in trainer.metrics
    assert 'precision' in trainer.metrics
    assert 'recall' in trainer.metrics
    assert 'f1_score' in trainer.metrics
