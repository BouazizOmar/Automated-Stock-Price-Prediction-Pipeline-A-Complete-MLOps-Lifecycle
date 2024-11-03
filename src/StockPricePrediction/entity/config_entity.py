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
    

    
    

