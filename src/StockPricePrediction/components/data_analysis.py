import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class DataAnalysis:
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def load_data(self):
        """Load processed data from CSV"""
        try:
            df = pd.read_csv(self.csv_file)
            df.index = pd.to_datetime(df.index)
            logging.info(f"Data loaded from {self.csv_file}")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return None

    def plot_stock_data(self, df):
        """Plot stock close prices and moving averages"""
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['close'], label='Close Price', color='blue')
        plt.plot(df.index, df['SMA_20'], label='20-period SMA', color='orange', linestyle='--')
        plt.plot(df.index, df['EMA_20'], label='20-period EMA', color='green', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Stock Price with Moving Averages')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_volume(self, df):
        """Plot trading volume over time"""
        plt.figure(figsize=(12, 6))
        sns.barplot(x=df.index, y=df['volume'], color='purple')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.title('Stock Trading Volume Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
