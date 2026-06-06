"""Shared utilities for the urban green cover project."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load project configuration from YAML."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def ensure_dirs(config: dict[str, Any]) -> dict[str, Path]:
    """Create output directories and return resolved paths."""
    paths = {
        "models": Path(config["output"]["models_dir"]),
        "figures": Path(config["output"]["figures_dir"]),
        "reports": Path(config["output"]["reports_dir"]),
        "logs": Path(config["output"]["logs_dir"]),
        "processed": Path(config["data"]["processed_dir"]),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save dictionary as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def class_names(config: dict[str, Any]) -> list[str]:
    """Return ordered class labels."""
    return list(config["data"]["classes"])
