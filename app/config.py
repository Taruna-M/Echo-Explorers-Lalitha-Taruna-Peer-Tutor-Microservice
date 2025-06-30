from functools import lru_cache
from pathlib import Path
import json

CONFIG_PATH = Path("config.json")
@lru_cache()
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

