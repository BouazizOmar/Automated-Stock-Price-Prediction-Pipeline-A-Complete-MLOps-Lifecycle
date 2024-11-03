from src.StockPricePrediction.components.data_preprocessing import DataPreprocessing
from src.StockPricePrediction.config.configuration import ConfigurationManager
from src.StockPricePrediction.constants import *
from src.StockPricePrediction import logger


STAGE_NAME = "Data Preprocessing Stage"

class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config_manager = ConfigurationManager(config_filepath=CONFIG_FILE_PATH)
        data_preprocessing_config = config_manager.get_data_preprocessing_config()

        data_preprocessing = DataPreprocessing(data_preprocessing_config)
        data_preprocessing.preprocess()
        


if __name__ == "__main__":
    try:
        logger.info(f">> Stage {STAGE_NAME} Started <<")
        obj = DataPreprocessingTrainingPipeline()
        obj.main()
        logger.info(f">> Stage {STAGE_NAME} Completed <<")
    except Exception as e:
        logger.exception(e)
        raise e
        
        