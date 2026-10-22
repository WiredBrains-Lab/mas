import json
import os

# Load the configuration file
config_file = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_file, 'r', encoding='utf-8') as f:
    config = json.load(f)