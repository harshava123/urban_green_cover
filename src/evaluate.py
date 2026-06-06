"""Model evaluation: metrics, confusion matrix, ROC-AUC, inference time."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_loader import create_datasets
from src.model_io import load_trained_model
from src.utils import class_names, ensure_dirs, load_config, save_json, set_seed


def _collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return integer labels and probability vectors."""
    y_true, y_prob = [], []

    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_prob.append(probs)
        y_true.append(np.argmax(labels.numpy(), axis=1))

    return np.concatenate(y_true), np.concatenate(y_prob)


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str,
) -> None:
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _measure_inference_time(model: tf.keras.Model, dataset: tf.data.Dataset, runs: int = 3) -> dict:
    """Measure average per-image inference latency."""
    sample_batch = next(iter(dataset))
    images = sample_batch[0]
    batch_size = images.shape[0]

    # Warm-up
    model.predict(images[:1], verbose=0)

    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(images, verbose=0)
        timings.append((time.perf_counter() - start) / batch_size)

    return {
        "avg_inference_ms_per_image": float(np.mean(timings) * 1000),
        "std_inference_ms_per_image": float(np.std(timings) * 1000),
        "batch_size": int(batch_size),
    }


def evaluate_model(model_name: str, config: dict | None = None) -> dict:
    """Evaluate one trained model on the held-out test set."""
    config = config or load_config()
    set_seed(config["project"]["random_seed"])
    paths = ensure_dirs(config)
    labels = class_names(config)

    model = load_trained_model(model_name, config=config)
    _, _, test_ds, _ = create_datasets(config, augment_train=False)

    y_true, y_prob = _collect_predictions(model, test_ds)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    try:
        metrics["roc_auc_ovr"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["roc_auc_ovr"] = None

    if config["evaluation"]["measure_inference_time"]:
        metrics.update(_measure_inference_time(model, test_ds))

    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)

    result = {
        "model_name": model_name,
        "metrics": metrics,
        "classification_report": report,
    }
    save_json(result, paths["reports"] / f"{model_name}_evaluation.json")

    if config["evaluation"]["save_confusion_matrix"]:
        _plot_confusion_matrix(
            y_true,
            y_pred,
            labels,
            paths["figures"] / f"{model_name}_confusion_matrix.png",
            title=f"{model_name} - Confusion Matrix",
        )

    print(f"\n{model_name} evaluation:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    return result


def evaluate_all(config: dict | None = None) -> list[dict]:
    """Evaluate all trained models and produce a comparison table."""
    config = config or load_config()
    paths = ensure_dirs(config)

    model_names = ["custom_cnn"] + list(config["models"]["transfer_learning"])
    results = []

    for name in model_names:
        weights_path = paths["models"] / f"{name}.weights.h5"
        legacy_path = paths["models"] / f"{name}.keras"
        if weights_path.exists() or legacy_path.exists():
            results.append(evaluate_model(name, config=config))

    if results:
        comparison = {
            r["model_name"]: r["metrics"] for r in results
        }
        save_json(comparison, paths["reports"] / "model_comparison.json")

    return results


if __name__ == "__main__":
    evaluate_all()
