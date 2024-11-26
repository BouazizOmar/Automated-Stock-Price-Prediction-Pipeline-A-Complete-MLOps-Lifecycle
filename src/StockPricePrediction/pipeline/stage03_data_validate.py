from src.StockPricePrediction.components.data_validate import DataValidation
from src.StockPricePrediction.config.configuration import ConfigurationManager
from src.StockPricePrediction.constants import *
from src.StockPricePrediction import logger


STAGE_NAME = "Data Validation Stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager(config_filepath=CONFIG_FILE_PATH)
        data_validation_config = config_manager.get_data_validation_config()

        data_validation = DataValidation(data_validation_config)
        if not data_validation.run():
            raise ValueError("Data validation failed.")

if __name__ == "__main__":
    try:
        logger.info(f">> Stage {STAGE_NAME} Started <<")
        pipeline = DataValidationPipeline()
        pipeline.main()
        logger.info(f">> Stage {STAGE_NAME} Completed <<")
    except Exception as e:
        logger.exception(e)
        raise e
