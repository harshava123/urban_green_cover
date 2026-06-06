"""Failure pattern analysis for deployment-oriented discussion (RQ5)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_loader import create_datasets
from src.model_io import load_trained_model
from src.utils import class_names, ensure_dirs, load_config, save_json, set_seed


def analyze_errors(model_name: str, config: dict | None = None) -> dict:
    """Summarize confusion patterns and high-confidence mistakes."""
    config = config or load_config()
    set_seed(config["project"]["random_seed"])
    paths = ensure_dirs(config)
    labels = class_names(config)

    model = load_trained_model(model_name, config=config)
    _, _, test_ds, _ = create_datasets(config, augment_train=False)

    rows = []
    for images, y_true_batch in test_ds:
        probs = model.predict(images, verbose=0)
        y_true = np.argmax(y_true_batch.numpy(), axis=1)
        y_pred = np.argmax(probs, axis=1)
        conf = np.max(probs, axis=1)

        for i in range(images.shape[0]):
            rows.append(
                {
                    "true_class": labels[int(y_true[i])],
                    "pred_class": labels[int(y_pred[i])],
                    "confidence": float(conf[i]),
                    "correct": bool(y_true[i] == y_pred[i]),
                }
            )

    df = pd.DataFrame(rows)
    errors = df[~df["correct"]]

    pair_counts = Counter(
        zip(errors["true_class"], errors["pred_class"])
    )
    top_confident_errors = errors.sort_values("confidence", ascending=False).head(10)

    summary = {
        "model_name": model_name,
        "total_predictions": len(df),
        "error_rate": float(1 - df["correct"].mean()),
        "most_common_confusion_pairs": [
            {"true": t, "predicted": p, "count": c}
            for (t, p), c in pair_counts.most_common(10)
        ],
        "high_confidence_errors": top_confident_errors.to_dict(orient="records"),
        "deployment_notes": {
            "screening_role": (
                "Use as decision-support for human review, not autonomous policy decisions."
            ),
            "main_risks": [
                "Adjacent greenery classes (sparse vs moderate) are visually similar.",
                "Seasonal and lighting variation may reduce robustness.",
                "Dataset geography may not generalize to new cities.",
                "High-confidence errors can mislead non-expert reviewers.",
            ],
            "validation_requirements": [
                "Prospective validation on locally captured street-level images.",
                "Expert adjudication for borderline cases.",
                "Periodic retraining with domain-shift monitoring.",
            ],
        },
    }

    save_json(summary, paths["reports"] / f"{model_name}_error_analysis.json")
    errors.to_csv(paths["reports"] / f"{model_name}_errors.csv", index=False)

    print(f"Error analysis complete for {model_name}")
    print(f"Error rate: {summary['error_rate']:.2%}")
    return summary


if __name__ == "__main__":
    cfg = load_config()
    for model in ["custom_cnn"] + cfg["models"]["transfer_learning"]:
        if (Path(cfg["output"]["models_dir"]) / f"{model}.keras").exists():
            analyze_errors(model, cfg)
