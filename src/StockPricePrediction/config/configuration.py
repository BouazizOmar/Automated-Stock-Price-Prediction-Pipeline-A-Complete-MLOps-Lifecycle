from src.StockPricePrediction.utils.common import read_yaml, create_directories
from src.StockPricePrediction.entity.config_entity import DataIngestionConfig
from src.StockPricePrediction.entity.config_entity import DataPreprocessingConfig

from dotenv import load_dotenv
from pathlib import Path
import os


# Load environment variables
class ConfigurationManager:
    def __init__(self, config_filepath: Path):
        self.config = read_yaml(config_filepath)
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config['data_collection']  
        
        # Create DataIngestionConfig using values from config.yaml
        data_ingestion_config = DataIngestionConfig(
            symbol=self.config['data_collection']['stock_symbols'],  
            interval=self.config['data_collection']['interval'],  
            outputsize=self.config['data_collection']['outputsize'], 
            api_key=os.getenv('ALPHA_VANTAGE_API_KEY'),  
            base_url=config['api_endpoint'],  
            output_dir=Path(self.config['data_paths']['raw_data']).parent  
        )
        
        create_directories([data_ingestion_config.output_dir])
        
        return data_ingestion_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config['preprocessing']
        
        # Create DataPreprocessingConfig using values from config.yaml 
        data_preprocessing_config = DataPreprocessingConfig(
            input_csv_file=Path(self.config['data_paths']['raw_data']),
            ema_window=self.config['preprocessing']['ema_window'],
            sma_window=self.config['preprocessing']['sma_window'],
            output_dir=Path(self.config['data_paths']['processed_data'])
        )
        
        create_directories([data_preprocessing_config.output_dir])
        
        return data_preprocessing_config