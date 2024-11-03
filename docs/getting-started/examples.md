# Ejemplos básico de uso

### Carga y despliegue de los datos de entrada originales

from turkish_music_emotion.dataset import DataHandler

df = dh.load_data('/data/raw/Acoustic Features.csv')
df

### Seleccion de Caracteristicas

from turkish_music_emotion.feature_selection import FeatureSelector

selector = FeatureSelector(data_dir)
selected_features = selector.select_features()

### Evaluacion del modelo

from turkish_music_emotion.model_evaluation_utils import ModelEvaluationUtils

MODEL_URI = "http://24.144.69.175:5000/#/experiments/10/runs/cf6fecc5c83b4447809e1875a1669ff3"
DATA_PATH = "../data/processed/Acoustic_Features_Processed.csv"

evaluator = ModelEvaluationUtils(MODEL_URI, DATA_PATH)
