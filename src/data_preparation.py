"""Dataset cleaning, mapping, and split preparation."""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.utils import load_config, save_json, set_seed


def _file_hash(path: Path) -> str:
    """Compute MD5 hash for duplicate detection."""
    hasher = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_valid_image(path: Path) -> bool:
    """Check whether an image file can be opened and decoded."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False


def _normalize_uc_merced_folder(folder_name: str) -> str:
    """Normalize Kaggle UC Merced folder names (e.g. uc_forest -> forest)."""
    name = folder_name.lower().strip()
    if name.startswith("uc_"):
        return name[3:]
    return name


def _map_uc_merced_class(folder_name: str, mapping: dict[str, list[str]]) -> str | None:
    """Map a UC Merced class folder to a greenery taxonomy class."""
    source_class = _normalize_uc_merced_folder(folder_name)
    for greenery_class, source_classes in mapping.items():
        if source_class in source_classes:
            return greenery_class
    return None


def prepare_dataset(
    raw_dir: str | Path,
    processed_dir: str | Path,
    config: dict | None = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    """
    Clean raw images, remove duplicates/corrupt files, map labels, and split.

    Expects UC Merced layout:
    raw_dir/{class_name}/*.jpg  (e.g. uc_forest/, agricultural/)
    """
    config = config or load_config()
    set_seed(config["project"]["random_seed"])

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    classes = config["data"]["classes"]
    mapping = config["data"]["uc_merced_mapping"]

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    for split in ("train", "val", "test"):
        for cls in classes:
            (processed_dir / split / cls).mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    seen_hashes: set[str] = set()
    skipped = Counter()

    image_paths = sorted(
        p for p in raw_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    for src_path in tqdm(image_paths, desc="Cleaning and mapping images"):
        if not _is_valid_image(src_path):
            skipped["corrupt"] += 1
            continue

        file_hash = _file_hash(src_path)
        if file_hash in seen_hashes:
            skipped["duplicate"] += 1
            continue
        seen_hashes.add(file_hash)

        source_class = _normalize_uc_merced_folder(src_path.parent.name)
        greenery_class = _map_uc_merced_class(src_path.parent.name, mapping)
        if greenery_class is None:
            skipped["unmapped"] += 1
            continue

        records.append(
            {
                "source_path": str(src_path),
                "source_class": source_class,
                "greenery_class": greenery_class,
                "hash": file_hash,
            }
        )

    if not records:
        raise ValueError(
            f"No valid images found in {raw_dir}. "
            "Expected UC Merced folders like uc_forest/, agricultural/, etc."
        )

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=config["project"]["random_seed"]).reset_index(drop=True)

    n = len(df)
    test_n = int(n * test_ratio)
    val_n = int(n * val_ratio)
    train_n = n - test_n - val_n

    df["split"] = "train"
    df.loc[train_n : train_n + val_n - 1, "split"] = "val"
    df.loc[train_n + val_n :, "split"] = "test"

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Writing processed splits"):
        split = row["split"]
        cls = row["greenery_class"]
        dst = processed_dir / split / cls / f"{row['hash']}.jpg"

        with Image.open(row["source_path"]) as img:
            img.convert("RGB").save(dst, format="JPEG", quality=95)

    summary = {
        "total_raw_candidates": len(image_paths),
        "total_valid": len(df),
        "skipped": dict(skipped),
        "class_balance": df["greenery_class"].value_counts().to_dict(),
        "split_balance": {
            f"{split}/{cls}": int(count)
            for (split, cls), count in df.groupby(["split", "greenery_class"]).size().items()
        },
    }

    save_json(summary, processed_dir / "dataset_summary.json")
    df.to_csv(processed_dir / "manifest.csv", index=False)

    print("\nDataset preparation complete.")
    print(f"Valid images: {len(df)}")
    print(f"Skipped: {dict(skipped)}")
    print("Class balance:", summary["class_balance"])

    return df


if __name__ == "__main__":
    cfg = load_config()
    prepare_dataset(
        raw_dir=cfg["data"]["raw_dir"],
        processed_dir=cfg["data"]["processed_dir"],
        config=cfg,
        val_ratio=cfg["data"]["validation_split"],
        test_ratio=cfg["data"]["test_split"],
    )
