"""Grad-CAM explainability for correct and incorrect predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.data_loader import create_datasets
from src.model_io import load_trained_model
from src.utils import class_names, ensure_dirs, load_config, set_seed


def _find_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """Locate the last convolutional layer for Grad-CAM (supports nested backbones)."""
    last_conv = None

    def _search(layer_list: list) -> None:
        nonlocal last_conv
        for layer in layer_list:
            if isinstance(layer, tf.keras.Model):
                _search(layer.layers)
            elif getattr(layer, "output_shape", None) and len(layer.output_shape) == 4:
                last_conv = layer

    _search(model.layers)
    if last_conv is None:
        raise ValueError("No convolutional layer found for Grad-CAM.")
    return last_conv


def _get_gradcam_layer(model: tf.keras.Model, model_name: str, config: dict) -> tf.keras.layers.Layer:
    """Resolve the Grad-CAM target layer for a trained model."""
    layer_map = config.get("explainability", {}).get("grad_cam_layers", {})
    if model_name in layer_map:
        layer_name = layer_map[model_name]
        backbone = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model)), None)
        if backbone is not None:
            return backbone.get_layer(layer_name)
        return model.get_layer(layer_name)
    return _find_last_conv_layer(model)


def make_gradcam_model(model: tf.keras.Model, model_name: str, config: dict) -> tf.keras.Model:
    """Build a model that returns conv features and predictions."""
    conv_layer = _get_gradcam_layer(model, model_name, config)
    return tf.keras.Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output],
        name="gradcam_model",
    )


def compute_saliency_map(
    model: tf.keras.Model,
    image: np.ndarray,
    class_index: int,
) -> np.ndarray:
    """Input-gradient saliency fallback when nested Grad-CAM is unavailable."""
    img_tensor = tf.expand_dims(tf.constant(image, dtype=tf.float32), axis=0)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor, training=False)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, img_tensor)[0]
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)
    return (saliency / (tf.reduce_max(saliency) + 1e-8)).numpy()


def compute_heatmap(
    grad_model: tf.keras.Model,
    image: np.ndarray,
    class_index: int,
) -> np.ndarray:
    """Compute Grad-CAM heatmap for a single image."""
    img_tensor = tf.expand_dims(image, axis=0)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        tape.watch(conv_outputs)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("Grad-CAM gradients are None; check the target layer connection.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def _overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay heatmap on the original image."""
    import cv2

    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip(alpha * colored + (1 - alpha) * image, 0, 1)


def _save_triplet(
    image: np.ndarray,
    heatmap: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Save original, heatmap, and overlay side by side."""
    overlay = _overlay_heatmap(image, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    fig.suptitle(title)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def generate_gradcam_samples(
    model_name: str,
    config: dict | None = None,
) -> dict:
    """
    Generate Grad-CAM visualizations for:
    - correct predictions
    - false positives
    - false negatives
    """
    config = config or load_config()
    set_seed(config["project"]["random_seed"])
    paths = ensure_dirs(config)
    labels = class_names(config)

    model = load_trained_model(model_name, config=config)
    grad_model = None
    method_label = "Grad-CAM"
    try:
        grad_model = make_gradcam_model(model, model_name, config)
    except Exception:
        method_label = "Saliency"

    _, _, test_ds, _ = create_datasets(config, augment_train=False)
    n_per_category = config["explainability"]["num_samples_per_category"]

    categories = {
        "correct": [],
        "false_positive": [],
        "false_negative": [],
    }

    for images, y_true_batch in test_ds:
        probs = model.predict(images, verbose=0)
        y_true = np.argmax(y_true_batch.numpy(), axis=1)
        y_pred = np.argmax(probs, axis=1)

        for i in range(images.shape[0]):
            raw_image = images[i].numpy()
            display_image = raw_image / 255.0 if raw_image.max() > 1.5 else raw_image

            true_idx = int(y_true[i])
            pred_idx = int(y_pred[i])

            if true_idx == pred_idx and len(categories["correct"]) < n_per_category:
                categories["correct"].append(
                    (raw_image, display_image, true_idx, pred_idx, float(probs[i][pred_idx]))
                )
            elif true_idx != pred_idx:
                if len(categories["false_positive"]) < n_per_category:
                    categories["false_positive"].append(
                        (raw_image, display_image, true_idx, pred_idx, float(probs[i][pred_idx]))
                    )
                if len(categories["false_negative"]) < n_per_category:
                    categories["false_negative"].append(
                        (raw_image, display_image, true_idx, pred_idx, float(probs[i][pred_idx]))
                    )

        if all(len(v) >= n_per_category for v in categories.values()):
            break

    out_dir = paths["figures"] / "grad_cam" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = {key: 0 for key in categories}
    for category, samples in categories.items():
        for idx, (raw_image, display_image, true_idx, pred_idx, confidence) in enumerate(samples):
            if grad_model is not None:
                try:
                    heatmap = compute_heatmap(grad_model, raw_image, pred_idx)
                except Exception:
                    heatmap = compute_saliency_map(model, raw_image, pred_idx)
                    method_label = "Saliency"
            else:
                heatmap = compute_saliency_map(model, raw_image, pred_idx)

            title = (
                f"{method_label} | {category} | true={labels[true_idx]} | "
                f"pred={labels[pred_idx]} | conf={confidence:.2f}"
            )
            _save_triplet(display_image, heatmap, title, out_dir / f"{category}_{idx+1}.png")
            saved[category] += 1

    summary = {"model_name": model_name, "saved": saved, "output_dir": str(out_dir)}
    print(f"Grad-CAM saved to {out_dir}")
    return summary


if __name__ == "__main__":
    cfg = load_config()
    for model in ["custom_cnn"] + cfg["models"]["transfer_learning"]:
        weights_path = Path(cfg["output"]["models_dir"]) / f"{model}.weights.h5"
        if weights_path.exists():
            generate_gradcam_samples(model, cfg)
