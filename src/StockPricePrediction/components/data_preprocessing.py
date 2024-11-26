import pandas as pd
import os
from src.StockPricePrediction import logger
from src.StockPricePrediction.entity.config_entity import DataPreprocessingConfig


class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def load_data(self):
        try:
            df = pd.read_csv(self.config.input_csv_file / "IBM_stock_data.csv")
            logger.info(f"Data loaded from {self.config.input_csv_file}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

    def apply_transformations(self, df):
        try:
            if 'Unnamed: 0' in df.columns:
                df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                logger.info("Renamed 'Unnamed: 0' to 'date' and set as index")
            
            # Z-score normalization
            df['close_normalized'] = (df['close'] - df['close'].mean()) / df['close'].std()
            
            
            # 20-period SMA and EMA
            df['SMA_20'] = df['close'].rolling(window=self.config.sma_window).mean()
            df['EMA_20'] = df['close'].ewm(span=self.config.ema_window, adjust=False).mean()
            
            # Lagged values
            df['close_lag1'] = df['close'].shift(1)
            df['close_lag3'] = df['close'].shift(3)
            
            # Filling missing values
            df['SMA_20'] = df['SMA_20'].fillna(df['SMA_20'].mean())
            df['close_lag1'] = df['close_lag1'].fillna(df['close_lag1'].mean())
            df['close_lag3'] = df['close_lag3'].fillna(df['close_lag3'].mean())
            

            
            logger.info("Applied transformations to data")
            return df
        except Exception as e:
            logger.error(f"Error applying transformations: {e}")
            return None

    def save_processed_data(self, df):
        try:
            os.makedirs(self.config.output_dir, exist_ok=True)
            processed_csv_path = self.config.output_dir / "processed_stock_data.csv"
            
            df.to_csv(processed_csv_path)
            
            logger.info(f"Processed data saved to {processed_csv_path}")
            return processed_csv_path
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            return None

    def preprocess(self):
        df = self.load_data()
        if df is not None:
            df = self.apply_transformations(df)
            if df is not None:
                self.save_processed_data(df)
