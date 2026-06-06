#!/usr/bin/env python3
"""End-to-end pipeline for Urban Green Cover Image Assessment."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_preparation import prepare_dataset
from src.error_analysis import analyze_errors
from src.evaluate import evaluate_all, evaluate_model
from src.grad_cam import generate_gradcam_samples
from src.train import train_all, train_model
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Urban Green Cover CNN pipeline (Idea 15)"
    )
    parser.add_argument(
        "--step",
        choices=["prepare", "train", "evaluate", "explain", "errors", "all"],
        default="all",
        help="Pipeline step to run",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Train/evaluate/explain a single model instead of all",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    if args.step in ("prepare", "all"):
        raw_dir = Path(config["data"]["raw_dir"])
        if not raw_dir.exists() or not any(raw_dir.rglob("*.jpg")):
            print(
                "WARNING: No images found in data/raw/.\n"
                "Download CSU-RSISC10 patches (see README) and place them in data/raw/."
            )
        else:
            prepare_dataset(
                raw_dir=config["data"]["raw_dir"],
                processed_dir=config["data"]["processed_dir"],
                config=config,
            )

    if args.step in ("train", "all"):
        if args.model:
            train_model(args.model, config=config)
        else:
            train_all(config=config)

    if args.step in ("evaluate", "all"):
        if args.model:
            evaluate_model(args.model, config=config)
        else:
            evaluate_all(config=config)

    if args.step in ("explain", "all"):
        models = [args.model] if args.model else ["custom_cnn"] + config["models"]["transfer_learning"]
        for name in models:
            models_dir = Path(config["output"]["models_dir"])
            if (models_dir / f"{name}.weights.h5").exists() or (models_dir / f"{name}.keras").exists():
                try:
                    generate_gradcam_samples(name, config=config)
                except Exception as exc:
                    print(f"Grad-CAM skipped for {name}: {exc}")

    if args.step in ("errors", "all"):
        models = [args.model] if args.model else ["custom_cnn"] + config["models"]["transfer_learning"]
        for name in models:
            models_dir = Path(config["output"]["models_dir"])
            if (models_dir / f"{name}.weights.h5").exists() or (models_dir / f"{name}.keras").exists():
                analyze_errors(name, config=config)

    print("\nPipeline step(s) finished.")


if __name__ == "__main__":
    main()
