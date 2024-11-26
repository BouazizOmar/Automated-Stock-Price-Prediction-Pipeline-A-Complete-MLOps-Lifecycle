from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    symbol: str
    interval: str
    outputsize: str
    api_key: str
    base_url: str
    output_dir: Path
    
    
@dataclass(frozen=True)
class DataPreprocessingConfig:
    input_csv_file: Path
    output_dir: Path
    sma_window: str
    ema_window: str


@dataclass(frozen=True)
class DataValidationConfig:
    preprocessed_data: Path
    required_columns: list
    checked_columns: list
    

@dataclass(frozen=True)
class ModelTrainingConfig:
    data_path: Path
    learning_rate: float
    epochs: int
    batch_size: int
    lstm_units: list
    scaler_path: Path
    dense_units: list
    
    
@dataclass(frozen=True)
class ModelEvaluationConfig:
    model_path: Path
    sequence_length: int
    mlflow_uri: str
    symbol: str
    interval: str
    outputsize: str
    base_url: str
    api_key: str

    
