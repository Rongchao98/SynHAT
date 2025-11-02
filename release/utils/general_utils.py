import json
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import yaml


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config_from_yaml(yaml_path: str | Path) -> SimpleNamespace:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return _to_namespace(yaml.safe_load(f))


def load_config_from_json(json_path: str | Path) -> SimpleNamespace:
    with open(json_path, "r", encoding="utf-8") as f:
        return _to_namespace(json.load(f))
