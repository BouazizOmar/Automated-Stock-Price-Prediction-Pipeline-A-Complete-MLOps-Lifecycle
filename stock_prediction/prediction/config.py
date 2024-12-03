import yaml

def load_config(path="stock_prediction/config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)

config = load_config()
