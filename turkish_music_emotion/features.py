from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from turkish_music_emotion.config import PROCESSED_DATA_DIR

app = typer.Typer()

class FeatureGenerator:
    def __init__(self, input_path: Path = PROCESSED_DATA_DIR / "dataset.csv", output_path: Path = PROCESSED_DATA_DIR / "features.csv"):
        self.input_path = input_path
        self.output_path = output_path

    def generate_features(self):
        """Simulate the feature generation process."""
        logger.info("Generating features from dataset...")
        for i in tqdm(range(10), total=10):
            if i == 5:
                logger.info("Something happened for iteration 5.")
        logger.success("Features generation complete.")

    def run(self):
        """Execute the feature generation process."""
        self.generate_features()

if __name__ == "__main__":
    feature_generator = FeatureGenerator()
    feature_generator.run()

