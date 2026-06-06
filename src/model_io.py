"""Reliable save/load for trained models."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from src.models import build_model
from src.utils import class_names, load_config


def save_trained_model(model: tf.keras.Model, model_name: str, models_dir: str | Path) -> str:
    """Save model weights (robust across nested transfer-learning backbones)."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    weights_path = models_dir / f"{model_name}.weights.h5"
    model.save_weights(str(weights_path))
    return str(weights_path)


def resolve_architecture_name(artifact_name: str, config: dict | None = None) -> str:
    """Map saved artifact names (e.g. ablation runs) to a buildable architecture."""
    config = config or load_config()
    known = {"custom_cnn"} | set(config["models"]["transfer_learning"])
    if artifact_name in known:
        return artifact_name
    for name in sorted(known, key=len, reverse=True):
        if artifact_name.startswith(name):
            return name
    return artifact_name


def load_trained_model(model_name: str, config: dict | None = None) -> tf.keras.Model:
    """Rebuild architecture and load saved weights."""
    config = config or load_config()
    labels = class_names(config)
    arch = resolve_architecture_name(model_name, config)
    model = build_model(arch, config, num_classes=len(labels))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config["training"]["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    weights_path = Path(config["output"]["models_dir"]) / f"{model_name}.weights.h5"
    legacy_path = Path(config["output"]["models_dir"]) / f"{model_name}.keras"

    if weights_path.exists():
        model.load_weights(str(weights_path))
    elif legacy_path.exists():
        legacy = tf.keras.models.load_model(str(legacy_path))
        model.set_weights(legacy.get_weights())
    else:
        raise FileNotFoundError(
            f"No saved weights for '{model_name}'. Train the model first."
        )

    return model
