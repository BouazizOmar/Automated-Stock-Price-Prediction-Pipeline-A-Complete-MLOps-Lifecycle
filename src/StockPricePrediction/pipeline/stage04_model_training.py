from src.StockPricePrediction.components.model_training import ModelTraining
from src.StockPricePrediction.config.configuration import ConfigurationManager
from src.StockPricePrediction.constants import *
from src.StockPricePrediction import logger

import os
import json

STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    def create_directories_if_not_exists(self, path: str):
        """Create directories if they don't already exist."""
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Created directory: {path}")
        else:
            logger.info(f"Directory already exists: {path}")

    def main(self):
        
        config_manager = ConfigurationManager(config_filepath=CONFIG_FILE_PATH)
        training_config = config_manager.get_model_training_config()

        model_training = ModelTraining(config=training_config)

        # Load and preprocess data
        data = model_training.load_data()
        dataset, scaled_data = model_training.scale_data(data)
        x_train, y_train, training_data_len = model_training.split_data(scaled_data)

        model_training.build_model()
        model_training.train_model(x_train, y_train)


        # Define the model and RMSE save paths
        model_save_path = os.path.join("artifacts", "model", "trained_model.h5")

        # Ensure that the directories for saving the model and RMSE exist
        self.create_directories_if_not_exists(os.path.dirname(model_save_path))

        # Save the trained model
        model_training.save_model(model_save_path)

    
   

if __name__ == "__main__":
    try:
        logger.info(f">> Stage {STAGE_NAME} Started <<")
        pipeline = ModelTrainingPipeline()
        pipeline.main()
        logger.info(f">> Stage {STAGE_NAME} Completed <<")
    except Exception as e:
        logger.exception(e)
        raise e
