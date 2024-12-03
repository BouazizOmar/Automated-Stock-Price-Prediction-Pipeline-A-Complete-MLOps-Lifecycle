import os
import json
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_from_path(model_path: str):
    """Load a trained model from a given path."""
    try:
        model = load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

def save_predictions_to_file(predictions: np.ndarray, file_path: str):
    """Save predictions to a file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savetxt(file_path, predictions, delimiter=',')
        logger.info(f"Predictions saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving predictions to {file_path}: {e}")
        raise

def scale_dataset(dataset: np.ndarray, feature_range=(0, 1)):
    """Scale the dataset using MinMaxScaler."""
    try:
        scaler = MinMaxScaler(feature_range=feature_range)
        scaled_data = scaler.fit_transform(dataset)
        logger.info("Dataset scaled successfully.")
        return scaled_data, scaler
    except Exception as e:
        logger.error(f"Error scaling dataset: {e}")
        raise

def create_sequences(data: np.ndarray, sequence_length: int):
    """Create sequences for time series data."""
    try:
        x, y = [], []
        for i in range(sequence_length, len(data)):
            x.append(data[i-sequence_length:i, 0])
            y.append(data[i, 0])
        x = np.array(x).reshape((-1, sequence_length, 1))
        y = np.array(y)
        logger.info("Sequences created successfully.")
        return x, y
    except Exception as e:
        logger.error(f"Error creating sequences: {e}")
        raise

def log_metrics_to_mlflow(metrics: dict, mlflow_uri: str, model):
    """Log metrics and model to MLflow."""
    import mlflow
    import mlflow.keras
    from urllib.parse import urlparse

    try:
        mlflow.set_registry_uri(mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(model, artifact_path="model")
            else:
                mlflow.keras.log_model(model, "model")

        logger.info("Metrics and model logged to MLflow successfully.")
    except Exception as e:
        logger.error(f"Error logging to MLflow: {e}")
        raise

def save_rmse_to_file(rmse: float, file_path: str):
    """Save RMSE to a file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(str(rmse))
        logger.info(f"RMSE saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving RMSE to {file_path}: {e}")
        raise

def fetch_real_time_data(config: dict) -> pd.DataFrame:
    """Fetch real-time stock data from an API."""
    import requests

    try:
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": config["symbol"],
            "interval": config["interval"],
            "apikey": config["api_key"],
            "datatype": "json"
        }
        response = requests.get(config["base_url"], params=params)
        response.raise_for_status()

        data = response.json()
        key = f"Time Series ({config['interval']})"

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
