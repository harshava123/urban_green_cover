#!/usr/bin/env python3
"""Generate training curves, ROC curves, and class distribution for the report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import create_datasets
from src.evaluate import _collect_predictions
from src.model_io import load_trained_model
from src.utils import class_names, ensure_dirs, load_config, set_seed


def plot_class_distribution(config: dict, out_dir: Path) -> None:
    """Bar chart of greenery class counts."""
    summary = json.loads(Path(config["data"]["processed_dir"]).joinpath("dataset_summary.json").read_text())
    counts = summary["class_balance"]
    df = pd.Series(counts).sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=df.index, y=df.values, palette="Greens_d")
    plt.title("Greenery Class Distribution (Processed Dataset)")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'class_distribution.png'}")


def plot_training_curves(logs_dir: Path, figures_dir: Path) -> None:
    """Loss and accuracy curves from CSV training logs."""
    for csv_path in sorted(logs_dir.glob("*_history.csv")):
        model_name = csv_path.stem.replace("_history", "")
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(df) + 1)

        axes[0].plot(epochs, df["loss"], label="Train")
        axes[0].plot(epochs, df["val_loss"], label="Validation")
        axes[0].set_title(f"{model_name} — Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, df["accuracy"], label="Train")
        axes[1].plot(epochs, df["val_accuracy"], label="Validation")
        axes[1].set_title(f"{model_name} — Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        fig.suptitle(f"Training Curves: {model_name}")
        plt.tight_layout()
        out_path = figures_dir / f"{model_name}_training_curves.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")


def plot_roc_curves(model_name: str, config: dict, figures_dir: Path) -> None:
    """One-vs-rest ROC curves for a trained model."""
    labels = class_names(config)
    model = load_trained_model(model_name, config=config)
    _, _, test_ds, _ = create_datasets(config, augment_train=False)
    y_true, y_prob = _collect_predictions(model, test_ds)

    y_bin = label_binarize(y_true, classes=list(range(len(labels))))

    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} — ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out_path = figures_dir / f"{model_name}_roc_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_model_comparison_table(config: dict, figures_dir: Path) -> None:
    """Save model comparison as a PNG table for the report."""
    comparison_path = Path(config["output"]["reports_dir"]) / "model_comparison.json"
    if not comparison_path.exists():
        return

    data = json.loads(comparison_path.read_text())
    rows = []
    for model, metrics in data.items():
        rows.append(
            {
                "Model": model,
                "Accuracy": f"{metrics['accuracy']:.3f}",
                "F1 (macro)": f"{metrics['f1_macro']:.3f}",
                "ROC-AUC": f"{metrics.get('roc_auc_ovr', 0):.3f}",
                "Inference (ms)": f"{metrics.get('avg_inference_ms_per_image', 0):.1f}",
            }
        )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 2 + 0.4 * len(df)))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)
    ax.set_title("Model Comparison (Test Set)", pad=20)
    plt.tight_layout()
    out_path = figures_dir / "model_comparison_table.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main() -> None:
    config = load_config()
    set_seed(config["project"]["random_seed"])
    paths = ensure_dirs(config)

    plot_class_distribution(config, paths["figures"])
    plot_training_curves(Path(config["output"]["logs_dir"]), paths["figures"])

    model_names = ["custom_cnn"] + list(config["models"]["transfer_learning"])
    for name in model_names:
        weights = paths["models"] / f"{name}.weights.h5"
        if weights.exists():
            plot_roc_curves(name, config, paths["figures"])

    plot_model_comparison_table(config, paths["figures"])
    print("\nReport figures generated.")


if __name__ == "__main__":
    main()
