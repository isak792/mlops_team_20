import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from turkish_music_emotion.dataset import DataHandler

@pytest.fixture
def real_data_structure():
    csv_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', 'Acoustic Features.csv')
    data = pd.read_csv(csv_file_path)
    metadata = Metadata.detect_from_dataframe(data)
    synthesizer = GaussianCopulaSynthesizer(metadata=metadata)
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(10)
    return synthetic_data

@pytest.fixture
def data_handler():
    return DataHandler()

@pytest.fixture
def temp_csv_path(tmp_path, real_data_structure):
    csv_path = tmp_path / "test_data.csv"
    real_data_structure.to_csv(csv_path, index=False)
    return csv_path

# # Se prueba la funcionalidad de la clase DataHandler para cargar datos
# def test_scale_data(data_handler, temp_csv_path):
#     data_handler.load_data(temp_csv_path.parent, temp_csv_path.name)
#     scaled_data = data_handler.scale_data()
#     assert scaled_data is not None
#     assert isinstance(scaled_data, pd.DataFrame)
#     assert np.allclose(scaled_data.mean(), 0, atol=1e-1)
#     assert np.allclose(scaled_data.std(), 1, atol=1e-1)

# Se prueba la funcionalidad de la clase DataHandler para cargar y procesar datos
def test_load_and_process_data(data_handler, temp_csv_path):
    loaded_data = data_handler.load_data(temp_csv_path.parent, temp_csv_path.name)
    assert loaded_data is not None
    assert isinstance(loaded_data, pd.DataFrame)
    
    data_handler.data.iloc[0, 1] = np.nan
    processed_data = data_handler.process_data()
    
    assert processed_data is not None
    assert not processed_data.isnull().any().any()
    assert isinstance(processed_data, pd.DataFrame)
