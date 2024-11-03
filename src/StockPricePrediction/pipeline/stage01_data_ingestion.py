from src.StockPricePrediction.config.configuration import ConfigurationManager
from src.StockPricePrediction.components.data_ingestion import DataIngestion
from src.StockPricePrediction.constants import *
from src.StockPricePrediction import logger



STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config_manager = ConfigurationManager(config_filepath=CONFIG_FILE_PATH) 
        data_ingestion_config = config_manager.get_data_ingestion_config()

        data_ingestion = DataIngestion(config=data_ingestion_config)
        stock_data = data_ingestion.fetch_stock_data()
        data_ingestion.save_raw_data_to_csv(stock_data)



if __name__ == "__main__":
    try:
        logger.info(f">>> Stage {STAGE_NAME} Started <<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>> Stage {STAGE_NAME} Completed<<<")
    except Exception as e:
        logger.exception(e)
        raise e