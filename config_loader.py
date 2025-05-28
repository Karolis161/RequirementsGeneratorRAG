import json
import os

CONFIG_PATH = "config.json"


def load_config(config_path=CONFIG_PATH):
    if not os.path.exists(config_path):
        print(f"Config file '{config_path}' not found. Using default values.")
        return {}

    try:
        with open(config_path, "r") as file:
            config = json.load(file)
        return config
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{config_path}'. Check for syntax errors.")
        return {}
    except Exception as e:
        print(f"Unexpected error loading config: {e}")
        return {}
