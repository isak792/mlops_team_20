import pytest
from pathlib import Path
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from turkish_music_emotion.dataset import DataHandler
from turkish_music_emotion.modeling.train import ModelTrainer
from turkish_music_emotion.modeling.predict import ModelPredictor

@pytest.fixture
def test_dirs(tmp_path):
    """Crear directorios temporales para pruebas"""
    dirs = {
        'raw': tmp_path / 'data' / 'raw',
        'processed': tmp_path / 'data' / 'processed',
        'models': tmp_path / 'models'
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    return dirs

@pytest.fixture
def synthetic_data(test_dirs):
    """Crear datos sintéticos basados en el dataset original"""
    csv_file_path = Path('data/raw/Acoustic Features.csv')
    real_data = pd.read_csv(csv_file_path)
    metadata = Metadata.detect_from_dataframe(real_data)
    synthesizer = GaussianCopulaSynthesizer(metadata=metadata)
    synthesizer.fit(real_data)
    synthetic_data = synthesizer.sample(10)
    
    synthetic_path = test_dirs['raw'] / 'AF_raw_test.csv'
    synthetic_data.to_csv(synthetic_path, index=False)
    return synthetic_path


@pytest.fixture
def patch_processed_dir(monkeypatch, test_dirs):
    """Cambiar la ruta de PROCESSED_DATA_DIR para pruebas"""
    monkeypatch.setattr('turkish_music_emotion.dataset.PROCESSED_DATA_DIR', test_dirs['processed'])
    yield


def test_pipeline_flow(test_dirs, synthetic_data, patch_processed_dir):
    """Prueba de integración para el flujo completo del pipeline"""
    params = {
        'data': {
            'raw_filename': 'AF_raw_test.csv',
            'processed_filename': 'AF_processed_test.csv'
        },
        'model': {
            'output_filename': 'model.joblib',
            'le_filename': 'label_encoder.joblib',
            'split_indices_filename': 'split_indices.json',
            'type': 'rf',
            'hyperparameters': {
                'rf': {'n_estimators': 10, 'max_depth': 3}
            }
        }
    }
    
    assert test_dirs['raw'].exists()
    assert test_dirs['processed'].exists()
    assert test_dirs['models'].exists()
    
    # Etapa 1: Cargar y procesar los datos
    data_handler = DataHandler()
    data_handler.run(
        test_dirs['raw'],
        params['data']['raw_filename'],
        params['data']['processed_filename']
    )
    
    processed_data = pd.read_csv(test_dirs['processed'] / params['data']['processed_filename'])
    
    assert synthetic_data.exists()
    synthetic_df = pd.read_csv(synthetic_data)
    assert not synthetic_df.empty
    assert 'Class' in processed_data.columns
    assert not processed_data.empty
    
    # Etapa 2: Entrenar el modelo
    X = processed_data.drop('Class', axis=1)
    trainer = ModelTrainer(
        test_dirs['processed'] / params['data']['processed_filename'],
        test_dirs['models'] / params['model']['output_filename'],
        test_dirs['models'] / params['model']['le_filename'],
        test_dirs['models'] / params['model']['split_indices_filename'],
        model_type=params['model']['type'],
        hyperparameters=params['model']['hyperparameters']
    )
    trainer.run()
    
    assert (test_dirs['models'] / params['model']['output_filename']).exists()
    assert (test_dirs['models'] / params['model']['le_filename']).exists()
    assert (test_dirs['models'] / params['model']['split_indices_filename']).exists()
    
    # Etapa 3: Hacer predicciones
    predictor = ModelPredictor(
        test_dirs['models'] / params['model']['output_filename'],
        test_dirs['models'] / params['model']['le_filename'],
        test_dirs['models'] / params['model']['split_indices_filename']
    )
    
    predictions = predictor.predict(X)
    
    assert len(predictions) > 0
    assert all(isinstance(pred, str) for pred in predictions)
    
    expected_classes = set(processed_data['Class'].unique())
    assert set(predictions).issubset(expected_classes)
