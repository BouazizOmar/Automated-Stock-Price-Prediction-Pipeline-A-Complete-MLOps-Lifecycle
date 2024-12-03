from django.http import JsonResponse
from django.views import View
from django.conf import settings
from src.StockPricePrediction.entity.config_entity import ModelEvaluationConfig
from src.StockPricePrediction.ModelEvaluator import ModelEvaluator

class ModelEvaluationView(View):
    def get(self, request, *args, **kwargs):
        """
        Handle GET requests for model evaluation.
        """
        try:
            # Fetch parameters from query
            symbol = request.GET.get("symbol", "AAPL")
            interval = request.GET.get("interval", "5min")
            
            # Define configuration
            config = ModelEvaluationConfig(
                symbol=symbol,
                interval=interval,
                api_key=settings.API_KEY,  # Add API key to settings
                base_url="https://www.alphavantage.co/query",
                model_path=settings.MODEL_PATH,  # Add model path to settings
                mlflow_uri=settings.MLFLOW_URI,  # Add MLflow URI to settings
                rmse_path=settings.RMSE_FILE_PATH,  # Add RMSE file path to settings
            )

            # Initialize ModelEvaluator
            evaluator = ModelEvaluator(config)

            # Fetch and preprocess real-time data
            scaled_data, dataset = evaluator.fetch_and_preprocess_real_time_data()

            # Test model on real-time data
            predictions = evaluator.test_model_real_time(scaled_data)


            return JsonResponse(response_data, status=200)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
