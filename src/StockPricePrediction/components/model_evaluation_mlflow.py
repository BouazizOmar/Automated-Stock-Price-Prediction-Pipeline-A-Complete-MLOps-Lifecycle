import os
import numpy as np
import pandas as pd
import requests  # Added missing import
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler  # Added missing import
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.keras
from urllib.parse import urlparse

from src.StockPricePrediction import logger
from src.StockPricePrediction.entity.config_entity import ModelEvaluationConfig


class ModelEvaluator:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.scaler = MinMaxScaler(feature_range=(0, 1))  
        
    def fetch_real_time_data(self) -> pd.DataFrame:
        try:
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": self.config.symbol,
                "interval": self.config.interval,
                "apikey": self.config.api_key,
                "datatype": "json"
            }
            response = requests.get(self.config.base_url, params=params)
            response.raise_for_status()

            data = response.json()
            key = f"Time Series ({self.config.interval})"
            if key not in data:
                raise ValueError(f"Unexpected response format: {data}")

            time_series = data[key]
            df = pd.DataFrame.from_dict(time_series, orient="index")
            df.columns = ["open", "high", "low", "close", "volume"]
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            logger.info("Real-time data fetched successfully.")
            return df
        except Exception as e:
            logger.error(f"Error fetching real-time data: {e}")
            raise

    def fetch_and_preprocess_real_time_data(self) -> np.ndarray:
        try:
            real_time_data = self.fetch_real_time_data()
            logger.info(f"Real-time data fetched for symbol: {self.config.symbol}")

            # Use the 'close' column for predictions
            dataset = real_time_data[['close']].values

            # Scale data
            scaled_data = self.scaler.fit_transform(dataset)  
            logger.info("Real-time data scaled.")
            return scaled_data, dataset
        except Exception as e:
            logger.error(f"Error fetching and preprocessing real-time data: {e}")
            raise

    def test_model_real_time(self, scaled_data: np.ndarray):
        try:
            x_real_time, _ = self._create_sequences(scaled_data)

            # Predict
            model = load_model(self.config.model_path) 
            predictions = model.predict(x_real_time)
            
            predictions = self.scaler.inverse_transform(predictions)

            logger.info("Real-time model testing complete.")
            return predictions
        except Exception as e:
            logger.error(f"Error during real-time model testing: {e}")
            raise
    
    def _create_sequences(self, data: np.ndarray):
        x, y = [], []
        for i in range(60, len(data)):
            x.append(data[i - 60:i, 0])
            y.append(data[i, 0])
        x = np.array(x).reshape((-1, 60, 1))
        y = np.array(y)
        return x, y

    def test_model(self, scaled_data: np.ndarray, training_data_len: int, dataset: np.ndarray):
        try:
            x_test, y_test = self._create_sequences(scaled_data)
            print(f"x_test shape: {x_test.shape}")

            model = load_model(self.config.model_path)  

            predictions = model.predict(x_test)
            predictions = self.scaler.inverse_transform(predictions)
            predictions = predictions.flatten()
            y_test = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            print(f"prediction mean: {predictions.mean()}")
            print(f"y_test mean: {y_test.mean()}")
            rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
            
            logger.info(f"Testing complete. RMSE: {rmse}")
            return rmse, predictions
        except Exception as e:
            logger.error(f"Error during model testing: {e}")
            raise
    
    
    
                
                
    def log_to_mlflow(self, rmse, predictions):        
        model = load_model(self.config.model_path)
        try:
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            print("Tracking URI: ", mlflow.get_tracking_uri())
            with mlflow.start_run():
                # Log metrics
                mlflow.log_metric("RMSE", rmse)
                # Log parameters
                if tracking_url_type_store != "file":
                    # track in the remote server
                    print("tracking url: ",tracking_url_type_store)
                    print("***remote***\n")
                    # Log model
                    mlflow.keras.log_model(model, artifact_path="model")
                else:
                    # track in the local
                    print("***local***\n")
                    mlflow.keras.log_model(model, "model")
        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
            raise

    def save_rmse(self, rmse: float):
        try:
            os.makedirs(os.path.dirname(self.config.rmse_path), exist_ok=True)
            with open(self.config.rmse_path, 'w') as f:
                f.write(str(rmse))
            logger.info(f"RMSE saved to {self.config.rmse_path}")
        except Exception as e:
            logger.error(f"Error saving RMSE: {e}")
            raise

