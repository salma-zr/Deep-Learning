from copy import deepcopy
from typing import Any, Dict

from .io import read_yaml


def load_config(path: str) -> Dict[str, Any]:
    cfg = read_yaml(path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a dict.")
    return cfg


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result
