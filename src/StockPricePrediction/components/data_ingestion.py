import requests
import pandas as pd
import logging
import os
import json

from src.StockPricePrediction.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def fetch_stock_data(self):
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': self.config.symbol,
            'interval': self.config.interval,
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

    def save_raw_data_to_csv(self, data):
        time_series_key = f'Time Series (1min)'
        if time_series_key in data:
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index = pd.to_datetime(df.index)
            df = df.apply(pd.to_numeric)

            csv_path = os.path.join(self.config.output_dir, f'{self.config.symbol}_stock_data.csv')
            df.to_csv(csv_path)
            logging.info(f"Raw data saved to {csv_path}")
            return csv_path
        else:
            logging.error("Time Series data not found in response")
            return None


