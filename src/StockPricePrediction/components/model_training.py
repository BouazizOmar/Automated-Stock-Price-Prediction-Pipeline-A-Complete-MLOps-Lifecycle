import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from src.StockPricePrediction import logger
from src.StockPricePrediction.entity.config_entity import ModelTrainingConfig
import joblib

class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    # Load data
    def load_data(self) -> pd.DataFrame:
        data_path = self.config.data_path
        try:
            data = pd.read_csv(data_path)
            logger.info(f"Data loaded from {data_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    # Scale data
    def scale_data(self, data: pd.DataFrame):
        try:
            dataset = data[['close']].values
            scaled_data = self.scaler.fit_transform(dataset)

            # Save the scaler to a file
            scaler_path = self.config.scaler_path  
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")

            logger.info("Data scaling complete.")
            return dataset, scaled_data
        except Exception as e:
            logger.error(f"Error during data scaling: {e}")
            raise

    # Split data
    def split_data(self, scaled_data: np.ndarray):
        try:
            training_data_len = len(scaled_data) - 100
            train_data = scaled_data[:training_data_len]

            x_train, y_train = self._create_sequences(train_data)
            logger.info("Data splitting complete.")
            return x_train, y_train, training_data_len
        except Exception as e:
            logger.error(f"Error during data splitting: {e}")
            raise

    # Helper to create sequences
    def _create_sequences(self, data: np.ndarray, sequence_length: int = 60):
        x, y = [], []
        for i in range(sequence_length, len(data)):
            x.append(data[i - sequence_length:i, 0])
            y.append(data[i, 0])
        x = np.array(x).reshape((-1, sequence_length, 1))
        y = np.array(y)
        return x, y

    # Build model
    def build_model(self):
        try:
            model = Sequential()

            # Add LSTM layers
            for units in self.config.lstm_units:
                model.add(LSTM(units, return_sequences=(units != self.config.lstm_units[-1])))

            # Add dense layers
            for units in self.config.dense_units:
                model.add(Dense(units))

            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')
            self.model = model
            logger.info("Model built successfully.")
        except Exception as e:
            logger.error(f"Error during model building: {e}")
            raise

    # Train model
    def train_model(self, x_train: np.ndarray, y_train: np.ndarray):
        try:
            self.model.fit(
                x_train, y_train, 
                batch_size=self.config.batch_size, 
                epochs=self.config.epochs
            )
            logger.info("Model training complete.")
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    # Save model
    def save_model(self, save_path: str):
        try:
            self.model.save(save_path)
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
