from src.StockPricePrediction.components.model_evaluation_mlflow import ModelEvaluator
from src.StockPricePrediction.config.configuration import ConfigurationManager
from src.StockPricePrediction.constants import *
from src.StockPricePrediction import logger

import os
import json

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass
    def main(self):
        
        config_manager = ConfigurationManager(config_filepath=CONFIG_FILE_PATH)
        evaluation_config = config_manager.get_model_evaluation_config()
        evaluator = ModelEvaluator(config=evaluation_config)
        data = evaluator.fetch_real_time_data()
        scaled_data, dataset = evaluator.fetch_and_preprocess_real_time_data()
        # Test model on real-time data
        logger.info("Testing model on real-time data...")
        predictions = evaluator.test_model_real_time(scaled_data)
        rmse, test_predictions = evaluator.test_model(scaled_data, len(scaled_data), dataset)
        logger.info("Logging results to MLflow...")
        try:
            evaluator.log_to_mlflow(rmse=rmse, predictions=test_predictions)
            logger.info("Logged to MLflow successfully.")
        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")

    
   

if __name__ == "__main__":
    try:
        logger.info(f">> Stage {STAGE_NAME} Started <<")
        pipeline = ModelEvaluationPipeline()
        pipeline.main()
        logger.info(f">> Stage {STAGE_NAME} Completed <<")
    except Exception as e:
        logger.exception(e)
        raise e
