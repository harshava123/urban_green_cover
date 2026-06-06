"""Training loop for custom CNN and transfer learning models."""

from __future__ import annotations

import time
from pathlib import Path

import tensorflow as tf

from src.data_loader import compute_class_weights, create_datasets
from src.model_io import save_trained_model
from src.models import build_model, model_param_count
from src.utils import class_names, ensure_dirs, load_config, save_json, set_seed


def _callbacks(config: dict, model_name: str, logs_dir: Path) -> list:
    """Create standard training callbacks."""
    train_cfg = config["training"]
    ckpt_path = str(logs_dir / f"{model_name}_best.weights.h5")

    return [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=train_cfg["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=train_cfg["reduce_lr_patience"],
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(logs_dir / f"{model_name}_history.csv")),
    ]


def train_model(
    model_name: str,
    config: dict | None = None,
    fine_tune: bool = False,
    save_as: str | None = None,
    augment_train: bool | None = None,
    use_class_weight: bool | None = None,
) -> dict:
    """Train a single model and persist artifacts."""
    config = config or load_config()
    set_seed(config["project"]["random_seed"])
    paths = ensure_dirs(config)
    artifact_name = save_as or model_name

    labels = class_names(config)
    aug = config["training"].get("augment_train", True) if augment_train is None else augment_train
    train_ds, val_ds, test_ds, _ = create_datasets(config, augment_train=aug)
    _ = test_ds  # used later in evaluation script

    model = build_model(model_name, config, num_classes=len(labels))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config["training"]["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    apply_weights = config["training"]["class_weight"] if use_class_weight is None else use_class_weight
    class_weight = compute_class_weights(config) if apply_weights else None

    start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["training"]["epochs"],
        class_weight=class_weight,
        callbacks=_callbacks(config, artifact_name, paths["logs"]),
        verbose=1,
    )
    train_time = time.perf_counter() - start

    final_path = save_trained_model(model, artifact_name, paths["models"])

    summary = {
        "model_name": artifact_name,
        "base_architecture": model_name,
        "parameters": model_param_count(model),
        "trainable_parameters": int(sum(tf.size(w).numpy() for w in model.trainable_weights)),
        "epochs_run": len(history.history["loss"]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "training_time_sec": train_time,
        "augmentation_used": aug,
        "class_weight_used": class_weight is not None,
        "fine_tune": fine_tune,
        "model_path": str(final_path),
    }
    save_json(summary, paths["reports"] / f"{artifact_name}_training_summary.json")

    print(f"\n{artifact_name} training complete.")
    print(f"Best val accuracy: {summary['best_val_accuracy']:.4f}")
    print(f"Training time: {train_time:.1f}s")

    return summary


def train_all(config: dict | None = None) -> list[dict]:
    """Train custom CNN and all configured transfer learning models."""
    config = config or load_config()
    summaries = []

    model_names = ["custom_cnn"] + list(config["models"]["transfer_learning"])
    for name in model_names:
        print(f"\n{'=' * 60}\nTraining: {name}\n{'=' * 60}")
        summaries.append(train_model(name, config=config))

    save_json({"models": summaries}, Path(config["output"]["reports_dir"]) / "all_training_summaries.json")
    return summaries


if __name__ == "__main__":
    train_all()
