import argparse
from pathlib import Path
from turkish_music_emotion.dataset import DataHandler
from turkish_music_emotion.modeling.train import ModelTrainer
from turkish_music_emotion.modeling.predict import ModelPredictor, load_data
from turkish_music_emotion.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
import yaml

def load_config(config_path: Path = Path('params.yaml')) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_process_data(params: dict):
    data_handler = DataHandler()
    data_handler.run(
        RAW_DATA_DIR,
        params['data']['raw_filename'],
        params['data']['processed_filename']
    )

def train_model(params: dict):
    trainer = ModelTrainer(
        PROCESSED_DATA_DIR / params['data']['processed_filename'],
        MODELS_DIR / params['model']['output_filename'],
        MODELS_DIR / params['model']['le_filename'],
        MODELS_DIR / params['model']['split_indices_filename'],
        model_type=params['model']['type'],
        hyperparameters=params['model']['hyperparameters']
    )
    trainer.run()

def predict(params: dict):
    predictor = ModelPredictor(
        MODELS_DIR / params['model']['output_filename'],
        MODELS_DIR / params['model']['le_filename'],
        MODELS_DIR / params['model']['split_indices_filename']
    )
    input_data = load_data(PROCESSED_DATA_DIR / params['data']['processed_filename'])
    predictions = predictor.predict(input_data)
    print(f"Predictions for test set: {predictions}")

def main():
    parser = argparse.ArgumentParser(description="Run the music emotion pipeline")
    parser.add_argument("--stage", 
                        choices=["load_process_data", "train_model", "predict"], 
                        help="Pipeline stage to run",
                        default="predict"  # Default stage when --stage is not provided
                        )
    args = parser.parse_args()

    params = load_config()

    if args.stage == "load_process_data":
        load_process_data(params)
    elif args.stage == "train_model":
        train_model(params)
    elif args.stage == "predict":
        predict(params)

if __name__ == "__main__":
    main()
