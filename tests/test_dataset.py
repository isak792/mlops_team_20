import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

# Define la ruta base una vez
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(BASE_DIR))

from turkish_music_emotion.dataset import DataHandler

@pytest.fixture
def real_data_structure():
    """Genera datos sintéticos basados en la estructura de datos reales."""
    csv_file_path = BASE_DIR / 'data' / 'raw' / 'Acoustic Features.csv'
    data = pd.read_csv(csv_file_path)
    metadata = Metadata.detect_from_dataframe(data)
    synthesizer = GaussianCopulaSynthesizer(metadata=metadata)
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(10)
    return synthetic_data

@pytest.fixture
def data_handler():
    """Instancia un DataHandler para pruebas de carga y procesamiento de datos."""
    return DataHandler()

@pytest.fixture
def temp_csv_path(tmp_path, real_data_structure):
    """Guarda los datos sintéticos en un archivo temporal para pruebas."""
    csv_path = tmp_path / "test_data.csv"
    real_data_structure.to_csv(csv_path, index=False)
    return csv_path

def test_load_and_process_data(data_handler, temp_csv_path):
    """Prueba la carga y el procesamiento de datos en DataHandler."""
    # Cargar los datos desde el archivo temporal
    loaded_data = data_handler.load_data(temp_csv_path.parent, temp_csv_path.name)
    assert loaded_data is not None, "Los datos no se cargaron correctamente."
    assert isinstance(loaded_data, pd.DataFrame), "Los datos cargados no son un DataFrame."
    
    data_handler.data = loaded_data
    # Introducir un valor nulo y procesar los datos
    loaded_data.iloc[0, 1] = np.nan
    processed_data =  data_handler.process_data()
    
    # Asegurar de que el procesamiento funcione correctamente
    assert processed_data is not None, "El procesamiento de datos falló."
    assert not processed_data.isnull().any().any(), "El procesamiento no eliminó los valores nulos."
    assert isinstance(processed_data, pd.DataFrame), "Los datos procesados no son un DataFrame."
    assert set(processed_data.columns) == set(loaded_data.columns), "La estructura de las columnas cambió tras el procesamiento."

def test_consistent_processing(data_handler, temp_csv_path):
    """Prueba que el procesamiento de datos sea consistente."""
    loaded_data = data_handler.load_data(temp_csv_path.parent, temp_csv_path.name)
    data_handler.data = loaded_data  # Asigna los datos cargados
    
    processed_data_first = data_handler.process_data()
    processed_data_second = data_handler.process_data()
    
    pd.testing.assert_frame_equal(processed_data_first, processed_data_second, 
                                  check_dtype=True, check_exact=True)

def test_process_data_with_extreme_values(data_handler, temp_csv_path):
    """Prueba que el procesamiento maneje valores extremos correctamente."""
    loaded_data = data_handler.load_data(temp_csv_path.parent, temp_csv_path.name)
    
    # Asigna el DataFrame cargado al atributo data de DataHandler
    data_handler.data = loaded_data
    
    # Introduce valores extremos en el DataFrame
    data_handler.data.iloc[0, :] = np.inf
    data_handler.data.iloc[1, :] = -np.inf
    
    processed_data = data_handler.process_data()  # Llama al método sin argumentos
    
    # Verifica que los valores extremos sean tratados o eliminados
    assert not processed_data.isin([np.inf, -np.inf]).any().any(), "El procesamiento no maneja valores extremos correctamente."

def test_load_empty_file(data_handler, tmp_path):
    """Prueba el comportamiento de carga con un archivo vacío."""
    empty_csv_path = tmp_path / "empty.csv"
    empty_csv_path.touch()  # Crea un archivo vacío
    
    with pytest.raises(ValueError):
        data_handler.load_data(empty_csv_path.parent, empty_csv_path.name)

def test_save_data(data_handler, temp_csv_path):
    """Prueba el guardado de datos en DataHandler."""
    loaded_data = data_handler.load_data(temp_csv_path.parent, temp_csv_path.name)
    data_handler.data = loaded_data
    
    temp_output_path = temp_csv_path.parent / "output_data.csv"
    data_handler.save_data(temp_csv_path.parent, "output_data.csv")
    
    saved_data = pd.read_csv(temp_output_path)
    pd.testing.assert_frame_equal(loaded_data, saved_data, check_dtype=True, check_exact=True)

def test_scale_data(data_handler, temp_csv_path):
    """Prueba el escalado de datos en DataHandler."""
    loaded_data = data_handler.load_data(temp_csv_path.parent, temp_csv_path.name)
    data_handler.data = loaded_data
    
    original_data = data_handler.data.copy()
    data_handler.scale_data()

    # Obtener columnas numéricas excluyendo aquellas con desviación estándar cero
    numeric_cols = data_handler.data.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = numeric_cols[data_handler.data[numeric_cols].std() != 0]

    # Aserciones
    assert not (data_handler.data == original_data).all().all(), "Los datos no se escalaron correctamente."
    
    if len(numeric_cols) > 0:  # Asegurarse de que haya columnas para verificar
        assert np.isclose(data_handler.data[numeric_cols].mean(), 0, atol=1e-1).all(), "La media de las columnas no es aproximadamente 0."
        assert np.isclose(data_handler.data[numeric_cols].std(), 1, atol=1e-1).all(), "La desviación estándar de las columnas no es aproximadamente 1."
    else:
        assert True, "No hay columnas numéricas válidas para verificar el escalado."









