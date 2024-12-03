import requests
import pandas as pd
from src.StockPricePrediction import logger
import os
import json

from src.StockPricePrediction.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def fetch_stock_data(self):
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': self.config.symbol,
            'apikey': self.config.api_key,
            'outputsize': self.config.outputsize
        }
        response = requests.get(self.config.base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            logging.error(f"Error fetching data: {response.status_code}")
            return None

    def save_raw_data_to_csv(self, json_data):
        time_series = json_data.get(f'Time Series (Daily)', {})


        if not time_series:
            logger.error(f"Time series data not found for interval {self.config.interval}")
            return None

        try:
            # Create DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index = pd.to_datetime(df.index)
            df = df.apply(pd.to_numeric)

            # Create directory if it does not exist
            os.makedirs(self.config.output_dir, exist_ok=True)

            # Save DataFrame to CSV
            csv_path = os.path.join(self.config.output_dir, f'{self.config.symbol}_stock_data.csv')
            df.to_csv(csv_path)
            logger.info(f"Raw data saved to {csv_path}")
            return csv_path
        except Exception as e:
            logger.error(f"Error saving data to CSV: {e}")
            return None


