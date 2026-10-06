"""JSON config loader: configs/default.json holds the paper's values."""
import json


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)

    def strip(d):
        return {k: strip(v) if isinstance(v, dict) else v for k, v in d.items() if not k.startswith("_")}
    return strip(cfg)
