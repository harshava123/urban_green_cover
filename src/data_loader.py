"""TensorFlow data pipelines with augmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from src.utils import class_names


def _build_augmentation(config: dict) -> tf.keras.Sequential:
    """Create training-time augmentation layers."""
    aug_cfg = config["augmentation"]
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal") if aug_cfg["horizontal_flip"] else tf.keras.layers.Lambda(lambda x: x),
            tf.keras.layers.RandomRotation(aug_cfg["rotation_range"] / 360.0),
            tf.keras.layers.RandomTranslation(
                aug_cfg["height_shift_range"],
                aug_cfg["width_shift_range"],
            ),
            tf.keras.layers.RandomZoom(aug_cfg["zoom_range"]),
        ],
        name="augmentation",
    )


def create_datasets(
    config: dict,
    augment_train: bool = True,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    Build train/val/test datasets from processed directory structure.

    Returns datasets and ordered class names.
    """
    processed_dir = Path(config["data"]["processed_dir"])
    image_size = tuple(config["data"]["image_size"])
    batch_size = config["data"]["batch_size"]
    labels = class_names(config)

    def _load_split(split: str, training: bool) -> tf.data.Dataset:
        ds = tf.keras.utils.image_dataset_from_directory(
            processed_dir / split,
            labels="inferred",
            label_mode="categorical",
            class_names=labels,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=training,
            seed=config["project"]["random_seed"],
        )
        # Keep 0-255 float32; each model applies its own rescaling/preprocessing.
        ds = ds.map(
            lambda x, y: (tf.cast(x, tf.float32), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if training and augment_train:
            augmentation = _build_augmentation(config)
            ds = ds.map(
                lambda x, y: (augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        return ds.prefetch(tf.data.AUTOTUNE)

    train_ds = _load_split("train", training=True)
    val_ds = _load_split("val", training=False)
    test_ds = _load_split("test", training=False)

    return train_ds, val_ds, test_ds, labels


def compute_class_weights(config: dict) -> dict[int, float]:
    """Compute balanced class weights from the training split manifest."""
    import pandas as pd
    from sklearn.utils.class_weight import compute_class_weight

    manifest = Path(config["data"]["processed_dir"]) / "manifest.csv"
    labels = class_names(config)

    if not manifest.exists():
        return {}

    df = pd.read_csv(manifest)
    train_df = df[df["split"] == "train"]
    y = train_df["greenery_class"].map({name: idx for idx, name in enumerate(labels)}).values

    weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    return {int(c): float(w) for c, w in zip(np.unique(y), weights)}
