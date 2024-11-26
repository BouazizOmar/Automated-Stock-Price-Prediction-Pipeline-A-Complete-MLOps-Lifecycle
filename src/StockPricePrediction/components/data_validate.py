import pandas as pd
from src.StockPricePrediction import logger
from src.StockPricePrediction.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.config.preprocessed_data / "processed_stock_data.csv")
            logger.info(f"Data loaded from {self.config.preprocessed_data}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise e

    def validate(self, df: pd.DataFrame) -> bool:
        try:
            # Check for missing required columns
            missing_columns = [
                col for col in self.config.required_columns if col not in df.columns
            ]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False

            # Check for missing values
            if df[self.config.checked_columns].isnull().any().any():
                logger.error("Data contains missing values.")
                return False

            logger.info("Validation passed. Data is correctly formatted.")
            return True

        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return False

    def run(self) -> bool:
    
        try:
            df = self.load_data()
            if self.validate(df):
                logger.info("Data validation pipeline completed successfully.")
                return True
            else:
                logger.error("Data validation failed.")
                return False
        except Exception as e:
            logger.error(f"Error running the validation pipeline: {e}")
            return False
