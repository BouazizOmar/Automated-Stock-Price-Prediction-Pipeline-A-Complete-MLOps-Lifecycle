import streamlit as st
from src.StockPricePrediction.components.model_evaluation_mlflow import ModelEvaluator
from src.StockPricePrediction.config.configuration import ConfigurationManager
from src.StockPricePrediction.constants import CONFIG_FILE_PATH
from src.StockPricePrediction import logger

# Initialize ConfigurationManager
config_manager = ConfigurationManager(config_filepath=CONFIG_FILE_PATH)
evaluation_config = config_manager.get_model_evaluation_config()

# Initialize Pipeline
evaluator = ModelEvaluator(config=evaluation_config)
data = evaluator.fetch_real_time_data()
scaled_data, dataset = evaluator.fetch_and_preprocess_real_time_data()


# Streamlit App
st.title("Stock Price Prediction with LSTM")
st.markdown("""
This application predicts the next stock price using a trained LSTM model. It also logs metrics and models to MLflow.
""")
rmse, test_predictions = evaluator.test_model(scaled_data, len(scaled_data), dataset)

# Display current stock price
st.subheader("Current Stock Price")
try:
    latest_close_price = data["close"].iloc[-1]
    st.write(f"**Current Stock Price ({evaluation_config.symbol}):** ${latest_close_price:.2f}")
except Exception as e:
    logger.error(f"Error fetching current stock price: {e}")
    st.error("Could not fetch the current stock price. Check the logs for details.")
    
    
# Predict next stock price
st.subheader("Predicted Next Stock Price")
try:
    scaled_data, dataset = evaluator.fetch_and_preprocess_real_time_data()
    predictions = evaluator.test_model_real_time(scaled_data)
    next_price = predictions[-1]  # Last prediction
    next_price = next_price[0]
    print(f"next_price: {next_price}")
    st.write(f"**Predicted Stock Price for Next Period:** ${next_price:.2f}")
except Exception as e:
    logger.error(f"Error predicting next stock price: {e}")
    st.error("Could not predict the next stock price. Check the logs for details.")

# Model Evaluation and Logging
st.subheader("Model Evaluation Metrics")
try:
    # Perform model evaluation
    rmse, test_predictions = evaluator.test_model(scaled_data, len(dataset), dataset)

    # Display RMSE
    st.write(f"**Root Mean Square Error (RMSE):** {rmse:.2f}")

    # Log metrics and model to MLflow
    evaluator.log_to_mlflow(rmse=rmse, predictions=test_predictions)
    st.success("Model metrics and artifact logged to MLflow successfully!")
except Exception as e:
    logger.error(f"Error during model evaluation and logging: {e}")
    st.error("Could not evaluate the model or log to MLflow. Check the logs for details.")
