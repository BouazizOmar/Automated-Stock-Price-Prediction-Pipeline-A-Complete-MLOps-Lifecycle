from src.StockPricePrediction.utils.common import read_yaml, create_directories
from src.StockPricePrediction.entity.config_entity import (
    DataIngestionConfig,
    DataPreprocessingConfig,
    DataValidationConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig,
)
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()  # Load environment variables


class ConfigurationManager:
    def __init__(self, config_filepath: Path):
        self.config = read_yaml(config_filepath)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config['data_collection']
        data_paths = self.config['data_paths']

        data_ingestion_config = DataIngestionConfig(
            symbol=config['stock_symbols'],
            interval=config['interval'],
            outputsize=config['outputsize'],
            api_key=os.getenv('ALPHA_VANTAGE_API_KEY'),
            base_url=config['api_endpoint'],
            output_dir=Path(data_paths['raw_data']),
        )

        create_directories([data_ingestion_config.output_dir])
        return data_ingestion_config

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config['preprocessing']
        data_paths = self.config['data_paths']

        data_preprocessing_config = DataPreprocessingConfig(
            input_csv_file=Path(data_paths['raw_data']),
            ema_window=config['ema_window'],
            sma_window=config['sma_window'],
            output_dir=Path(data_paths['processed_data']),
        )

        create_directories([data_preprocessing_config.output_dir])
        return data_preprocessing_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config['data_validation']
        data_paths = self.config['data_paths']

        data_validation_config = DataValidationConfig(
            preprocessed_data=Path(data_paths['processed_data']),
            required_columns=config['required_columns'],
            checked_columns=config['checked_columns'],
        )

        return data_validation_config

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config['model_params']

        model_training_config = ModelTrainingConfig(
            data_path=Path(config['data_path']),
            learning_rate=config['learning_rate'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            lstm_units=config['lstm_units'],
            scaler_path=Path(config['scaler_path']),
            dense_units=config['dense_units'],
        )

        return model_training_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config['model_evaluation_params']
        
        model_evaluation_config = ModelEvaluationConfig(
            model_path=Path(config['model_path']),
            sequence_length=config['sequence_length'],
            mlflow_uri=config['mlflow_uri'],
            symbol=config['symbol'],
            interval=config['interval'],
            outputsize=config['outputsize'],
            base_url=config['api_endpoint'],
            api_key=os.getenv('ALPHA_VANTAGE_API_KEY'),

        )
        
        return model_evaluation_config