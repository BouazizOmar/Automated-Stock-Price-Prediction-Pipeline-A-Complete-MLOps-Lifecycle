from src.StockPricePrediction import logger
from src.StockPricePrediction.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
from src.StockPricePrediction.pipeline.stage02_data_preprocessing import DataPreprocessingTrainingPipeline
from src.StockPricePrediction.pipeline.stage03_data_validate import DataValidationTrainingPipeline
from src.StockPricePrediction.pipeline.stage04_model_training import ModelTrainingPipeline
from src.StockPricePrediction.pipeline.stage05_model_evaluation import ModelEvaluationPipeline


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

STAGE_NAME = "Data Validation Stage"

try: 
    logger.info(f">>> Stage {STAGE_NAME} Started<<<")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logger.info(f">>> Stage {STAGE_NAME} Completed<<<")
except Exception as e:
    raise e


STAGE_NAME = "Model Training Stage"

try: 
    logger.info(f">>> Stage {STAGE_NAME} Started<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>> Stage {STAGE_NAME} Completed<<<")
except Exception as e:
    raise e

STAGE_NAME = "Model Evaluation Stage"

try: 
    logger.info(f">>> Stage {STAGE_NAME} Started<<<")
    obj = ModelEvaluationPipeline()
    obj.main()
    logger.info(f">>> Stage {STAGE_NAME} Completed<<<")
except Exception as e:
    raise e