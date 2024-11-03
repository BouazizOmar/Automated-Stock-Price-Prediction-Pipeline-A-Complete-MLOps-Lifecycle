from src.StockPricePrediction import logger
from src.StockPricePrediction.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
from src.StockPricePrediction.pipeline.stage02_data_preprocessing import DataPreprocessingTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try: 
    logger.info(f">>> Stage {STAGE_NAME} Started<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>> Stage {STAGE_NAME} Completed<<<")
except Exception as e:
    raise e



STAGE_NAME = "Data Preprocessing Stage"

try: 
    logger.info(f">>> Stage {STAGE_NAME} Started<<<")
    obj = DataPreprocessingTrainingPipeline()
    obj.main()
    logger.info(f">>> Stage {STAGE_NAME} Completed<<<")
except Exception as e:
    raise e

