from turkish_music_emotion.dataset import DataHandler
from turkish_music_emotion.modeling.train import ModelTrainer
from turkish_music_emotion.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
import yaml

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

input_filename = params['data']['raw_filename']
output_filename = params['data']['processed_filename']
model_output_filename = params['model']['output_filename']
model_type = params['model']['type']  # Add this line to get the model type from config

def main():
    data_handler = DataHandler()   
    data_handler.run(RAW_DATA_DIR, input_filename, output_filename)
    
    model_trainer = ModelTrainer(
        PROCESSED_DATA_DIR / output_filename,
        MODELS_DIR / model_output_filename,
        model_type=model_type  # Pass the model type to ModelTrainer
    )
    model_trainer.run()

if __name__ == "__main__":
    main()
